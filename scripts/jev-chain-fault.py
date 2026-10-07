#!/usr/bin/env python3
"""Chain-walk fault profiler for semblr rounds (run 1).

Given a target round id, walk its parentId chain to the root and ask Jev
(via the OpenRouter Decisions API) for ONE number per round: the probability
that the round is faulty — that the assistant response mishandled the user
prompt compared to what a competent assistant should have done with the same
input and the same context.

Each round is judged statelessly: the payload contains the round's own
input, output (tool calls redacted), memory/search tool-call arguments, and
whether its parent exists. Later corrections are deliberately NOT shown —
they would bias every verdict toward "faulty" (hindsight bias).

Fault scores are anchored bands (0/25/50/75/100), not free-form probabilities:
  0   defensible given the payload
  25  mostly defensible
  50  cannot tell from the payload
  75  probably faulty
  100 clearly faulty

From the per-round scores the tool derives the origin distribution over the
chain (earliest-faulty-round semantics):
  P(origin = i) = f_i * prod_{j older than i} (1 - f_j)
  P(origin outside chain) = prod_j (1 - f_j)
Long chains of mediocre scores wash the posterior flat instead of confidently
picking a wrong origin.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python3 scripts/jev-chain-fault.py ROUND_ID [ROUND_ID ...]
      [--rounds-dir DIR] [--state-chars 12000] [--max-depth 0]
      [--out temp/jev-chain-fault.jsonl] [--dry-run] [--quiet]

Outputs:
  - one JSONL record per target round (chain, scores, origin profile)
  - a human-readable chain table on stdout

RUBRIC VERSION: fault-bands-v1 (frozen text — do not edit per run;
bump the version string if the rubric changes, so scores stay comparable).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
RUBRIC_VERSION = "fault-bands-v1"

FAULT_QUESTION = {
    "fault": {
        "type": "score",
        "instructions": (
            "Judge ONLY this round: the user prompt below and the assistant "
            "response to it. How likely is it that THIS round is faulty — "
            "that the assistant response mishandled this request compared to "
            "what a competent assistant should have done with the same input "
            "and the same context? Do not judge the user. Nothing about what "
            "happened later is available or relevant: a response that is "
            "reasonable given what the round knew is NOT faulty even if the "
            "conversation later went wrong. Fault means the response itself "
            "introduces or extends a mishap: wrong action, wrong claim, or "
            "failing to use what it already had in this payload."
        ),
        "criteria": [
            "0 — Defensible: the response is what a competent assistant would produce from this prompt and context.",
            "1 — Mostly defensible: minor weaknesses only; the request was not mishandled.",
            "2 — Cannot tell: the payload does not show whether the request was mishandled.",
            "3 — Probably faulty: the response likely mishandled or misstated the request despite having what it needs.",
            "4 — Faulty: the response clearly introduces or extends a mishap — wrong action, wrong claim, or failing to use what it already had.",
        ],
    },
}

SEARCH_TOOLS = ("search_interactions", "get_round_details", "get_tool_details")


def load_round(rounds_dir: Path, rid: str) -> dict | None:
    path = rounds_dir / f"{rid}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def response_text_only(round_data: dict) -> tuple[str, int]:
    texts = []
    n_tools = 0
    for seg in round_data.get("responseSegments") or []:
        if seg.get("type") == "toolCall":
            n_tools += 1
        elif seg.get("type") == "text" and seg.get("text"):
            texts.append(seg["text"])
    return "\n".join(texts), n_tools


def searched_calls(round_data: dict, arg_chars: int = 200) -> list[str]:
    lines = []
    for tc in round_data.get("toolCalls") or []:
        if tc.get("name") in SEARCH_TOOLS:
            args = (tc.get("arguments") or "").replace("\n", " ")
            lines.append(f"- {tc['name']} {args[:arg_chars]}")
    return lines


def build_state(round_data: dict, parent_exists: str, state_chars: int) -> str:
    prompt = round_data.get("userPrompt") or ""
    response, n_tools = response_text_only(round_data)
    redacted = ""
    if n_tools:
        redacted = f"\n[{n_tools} tool calls in the response were redacted]"
    searched = searched_calls(round_data)
    searched_block = "\n".join(searched) if searched else "- none"
    state = (
        f"USER PROMPT (the claim under review):\n{prompt}\n\n"
        f"ASSISTANT RESPONSE (tool calls redacted):\n{response}{redacted}\n\n"
        f"MEMORY TOOL CALLS THIS ROUND MADE (arguments only):\n{searched_block}\n\n"
        f"CONTEXT THE ROUND SAW (memory injection): not available — "
        f"injections were not persisted for this round; judge groundedness "
        f"only from the payload above.\n\n"
        f"PARENT ROUND EXISTS: {parent_exists}"
    )
    return state[:state_chars]


def query_jev(state: str, api_key: str) -> dict:
    payload = json.dumps({"model": MODEL, "state": state, "questions": FAULT_QUESTION}).encode()
    req = request.Request(
        DECISIONS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    answers = body.get("answers") or body.get("results") or {}
    if "fault" not in answers:
        raise ValueError(f"unexpected response shape: {list(body.keys())}")
    return answers["fault"]


def walk_chain(rounds_dir: Path, target: str, max_depth: int) -> list[dict]:
    """Return links newest-first: target, parent, ..., root."""
    links = []
    seen = set()
    rid = target
    while rid and rid not in seen:
        if max_depth and len(links) >= max_depth:
            break
        seen.add(rid)
        data = load_round(rounds_dir, rid)
        if data is None:
            links.append({"id": rid, "missing": True, "parent_exists": "unknown (round file missing)"})
            break
        parent_id = (data.get("parentId") or "").removesuffix(".json") or None
        links.append({
            "id": rid,
            "data": data,
            "parent_id": parent_id,
            "parent_exists": "yes" if parent_id and (rounds_dir / f"{parent_id}.json").exists() else ("yes" if parent_id else "no"),
        })
        rid = parent_id
    return links


def origin_profile(scores_oldest_first: list[float]) -> list[float]:
    """P(origin = i) per link + P(outside chain) as the last element."""
    probs = []
    clean_run = 1.0
    for f in scores_oldest_first:
        probs.append(f * clean_run)
        clean_run *= 1.0 - f
    probs.append(clean_run)  # outside chain
    return probs


def report(record: dict, quiet: bool) -> None:
    if quiet or record.get("dry_run"):
        return
    chain = record["chain"]
    print(f"\nCHAIN for {record['target']} (oldest first)")
    print(f"{'#':>2}  {'f':>4}  {'P(origin)':>9}  {'id':<34}  prompt")
    for i, link in enumerate(chain):
        prompt = link.get("prompt_preview", "")
        print(
            f"{i + 1:>2}  {link.get('f', float('nan')):>4.2f}  "
            f"{record['p_origin'][i]:>9.2f}  {link['id']:<34}  {prompt}"
        )
    print(f"     P(origin outside chain) = {record['p_origin'][-1]:.2f}")
    print(f"     entropy = {record['entropy']:.2f} nats, top = {record['top_origin']}")


def profile_target(rounds_dir: Path, target: str, api_key: str, args) -> dict:
    links = walk_chain(rounds_dir, target, args.max_depth)
    oldest_first = list(reversed(links))
    chain_out = []
    scores = []
    for link in oldest_first:
        if link.get("missing"):
            chain_out.append({"id": link["id"], "f": None, "error": "round file missing"})
            continue
        parent_exists = link["parent_exists"]
        state = build_state(link["data"], parent_exists, args.state_chars)
        if args.dry_run:
            print(f"===== state for {link['id']} ({len(state)} chars) =====")
            print(state)
            chain_out.append({"id": link["id"], "f": None, "state_chars": len(state)})
            continue
        fault = query_jev(state, api_key)
        raw_score = float(fault.get("score", 2.0))
        f = round(min(max(raw_score / 4.0, 0.0), 1.0), 4)
        scores.append(f)
        chain_out.append({
            "id": link["id"],
            "parent_id": link.get("parent_id"),
            "parent_exists": parent_exists,
            "prompt_preview": (link["data"].get("userPrompt") or "")[:100].replace("\n", " "),
            "f": f,
            "raw_score": raw_score,
        })

    if args.dry_run:
        return {"target": target, "dry_run": True, "chain": chain_out}

    p = origin_profile(scores)
    import math
    entropy = -sum(x * math.log(x) for x in p if x > 0)
    top_i = max(range(len(scores)), key=lambda i: p[i]) if scores else None
    top_origin = chain_out[top_i]["id"] if top_i is not None else None
    if p[-1] > (p[top_i] if top_i is not None else -1):
        top_origin = "outside_chain"
    return {
        "target": target,
        "rubric_version": RUBRIC_VERSION,
        "model": MODEL,
        "chain": chain_out,
        "p_origin": [round(x, 4) for x in p],
        "entropy": round(entropy, 3),
        "top_origin": top_origin,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="target round id(s) to profile")
    ap.add_argument("--rounds-dir", default=os.path.expanduser("~/.pi/agent/semblr/rounds"))
    ap.add_argument("--state-chars", type=int, default=12000)
    ap.add_argument("--max-depth", type=int, default=0, help="max chain links to walk (0 = to root)")
    ap.add_argument("--out", default="temp/jev-chain-fault.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print assembled states, no API calls")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1

    rounds_dir = Path(args.rounds_dir)
    if not rounds_dir.exists():
        print(f"ERROR: no rounds dir at {rounds_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    errors = 0
    with out_path.open("a") as out_f:
        for target in args.targets:
            try:
                record = profile_target(rounds_dir, target, api_key or "", args)
            except (error.HTTPError, error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors += 1
                print(f"ERROR {target}: {exc}", file=sys.stderr)
                continue
            if not args.dry_run:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
            report(record, args.quiet)

    if errors:
        print(f"\ndone with {errors} error(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
