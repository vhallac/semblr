#!/usr/bin/env python3
"""Staged fault-analysis pipeline for semblr rounds.

Stages (per round):
  S0  jev: round filter        -> fault score f in [0,1]; rank, do NOT threshold
  S1  jev: fault type          -> provisional type from the taxonomy (noul per type)
  S2  LLM: root cause analysis -> root cause narrative
                                  | "fault kind not observed"
                                  | "no fault within round"
  S3  LLM: fault re-type (only if S2 said "fault kind not observed")
                               -> corrected type | "no fault within round"
  S4  LLM: revised root cause (only if S3 corrected the type)

Bounded re-route: at most ONE S2->S3->S4 cycle per round.

Final bins:
  NO_FAULT            S2/S3 concluded no fault within the round
  EXTERNAL            root cause lies outside this round (provenance/injection)
  AMBIGUOUS           LLM declined to judge
  <fault-type>        per-taxonomy bins with root cause attached

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python3 scripts/fault-pipeline.py ROUND_ID [ROUND_ID ...]
      [--ids-file FILE] [--rounds-dir DIR] [--top-k 50] [--top-frac 0.25]
      [--stages S0,S1,S2] [--workers 4] [--llm-model z-ai/glm-5.3-flash]
      [--state-chars 12000] [--chain-chars 24000]
      [--out temp/fault-pipeline.jsonl] [--resume] [--dry-run]
      [--persist-eval] [--eval]

Outputs:
  - temp/fault-pipeline.jsonl  one record per round with stage-by-stage verdicts
  - temp/eval-set.json         frozen evaluation set (with --persist-eval)

jev is a recall-oriented filter: its scores RANK candidates, they never gate.
False positives at S0/S1 are acceptable by design; the LLM stages absorb them
via the first-class "no fault within round" terminal bin.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_MODEL = "typesafe/jev-1.13"
DEFAULT_LLM_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_ROUNDS_DIR = os.path.expanduser("~/.pi/agent/semblr/rounds")

TAXONOMY = [
    "provenance-injection",
    "steering-interference",
    "retrieval-noise",
    "stale-context",
    "tool-result-mangling",
    "prompt-misread",
    "other",
]
NO_FAULT = "no-fault-within-round"
NOT_OBSERVED = "fault-kind-not-observed"
EXTERNAL = "external"
AMBIGUOUS = "ambiguous"

FAULT_TYPE_DESCRIPTIONS = {
    "provenance-injection": (
        "The context injected from past conversations was wrong, stale, or "
        "misattributed, so the assistant built on a faulty premise about "
        "history."
    ),
    "steering-interference": (
        "Mid-inference steering messages or merged steering content derailed "
        "the response away from the original prompt."
    ),
    "retrieval-noise": (
        "The similarity-retrieval step surfaced irrelevant past rounds that "
        "polluted the context."
    ),
    "stale-context": (
        "The assistant relied on outdated facts, old file contents, or a "
        "superseded decision from earlier in the session."
    ),
    "tool-result-mangling": (
        "A tool call returned wrong, truncated, or misinterpreted results "
        "that the response then treated as truth."
    ),
    "prompt-misread": (
        "The assistant misunderstood or partially ignored what the user "
        "actually asked for, with clean context."
    ),
    "other": (
        "The fault is real but none of the listed categories fit."
    ),
}

FAULT_TYPE_QUESTIONS = {
    t: {
        "type": "noul",
        "instructions": (
            "Given this user prompt and assistant response, is the fault "
            f"best described as '{t}'? {FAULT_TYPE_DESCRIPTIONS[t]} "
            "Answer with the probability that this description fits."
        ),
    }
    for t in TAXONOMY
}

S2_SYSTEM = (
    "You are a root-cause analyst for an AI coding agent's conversation "
    "rounds. You are given a target round (user prompt + assistant response) "
    "and the parental prompt chain that led to it. Judge whether the "
    "assistant response mishandled the request, and if so, classify the root "
    "cause. Respond with ONLY a JSON object:\n"
    '{"verdict": "fault_observed" | "fault_kind_not_observed" | '
    '"no_fault_within_round" | "external" | "ambiguous",\n'
    ' "fault_type": one of the taxonomy values or null,\n'
    ' "root_cause": "concise narrative or null",\n'
    ' "evidence": "short quote or observation or null"}\n'
    "Semantics:\n"
    "- fault_observed: the response mishandled the request; give fault_type "
    "from the taxonomy and a root_cause narrative.\n"
    "- fault_kind_not_observed: something is off but the taxonomy does not "
    "describe it; leave fault_type null. A later stage will re-classify.\n"
    "- no_fault_within_round: the response is defensible given what it saw; "
    "any fault lies elsewhere or nowhere.\n"
    "- external: the root cause lies outside this round's text (e.g. wrong "
    "context was injected upstream); name the external mechanism in "
    "root_cause.\n"
    "- ambiguous: the payload is insufficient to judge.\n"
    f"Taxonomy: {', '.join(TAXONOMY)}."
)

S3_SYSTEM = (
    "You are re-classifying a conversation round. A first analysis said the "
    "round is faulty but could not name the fault kind. You are given the "
    "round, its parental prompt chain, and the first analysis. Decide the "
    "correct fault type from the taxonomy, or conclude "
    f"'{NO_FAULT}' if the round is actually defensible. Respond with ONLY "
    "a JSON object:\n"
    f'{{"fault_type": "<one of: {", ".join(TAXONOMY)}, {NO_FAULT}>",\n'
    ' "reason": "short justification"}}'
)

S4_SYSTEM = (
    "You are revising a root-cause analysis. The fault type was corrected by "
    "a later stage; rewrite the root cause so it is consistent with the "
    "corrected type. Respond with ONLY a JSON object:\n"
    '{"root_cause": "revised concise narrative",\n'
    ' "evidence": "short quote or observation"}'
)


def jev_request(state: str, api_key: str, questions: dict) -> dict:
    payload = json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}).encode()
    req = request.Request(
        DECISIONS_URL, data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    answers = body.get("answers") or body.get("results") or {}
    if not answers:
        raise ValueError(f"unexpected jev response shape: {list(body.keys())}")
    return answers


def llm_request(system: str, user: str, api_key: str, model: str, max_retries: int = 3) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
    }).encode()
    last_exc = None
    for attempt in range(max_retries):
        req = request.Request(
            CHAT_URL, data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except (error.HTTPError, error.URLError, KeyError, json.JSONDecodeError) as exc:
            last_exc = exc
            if isinstance(exc, error.HTTPError) and exc.code in (401, 403, 404):
                raise
            time.sleep(2 ** attempt)
    raise last_exc


def parse_json_loose(text: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in LLM response: {text[:200]}")
    return json.loads(text[start:end + 1])


# ---- round loading / state assembly (mirrors jev-chain-fault.py) ----

SEARCH_TOOLS = ("search_interactions", "get_round_details", "get_tool_details")


def load_round(rounds_dir: Path, rid: str) -> dict | None:
    path = rounds_dir / f"{rid}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def response_text_only(round_data: dict) -> tuple[str, int]:
    texts, n_tools = [], 0
    for seg in round_data.get("responseSegments") or []:
        if seg.get("type") == "toolCall":
            n_tools += 1
        elif seg.get("type") == "text" and seg.get("text"):
            texts.append(seg["text"])
    return "\n".join(texts), n_tools


def build_round_state(round_data: dict, parent_exists: str, state_chars: int) -> str:
    prompt = round_data.get("userPrompt") or ""
    response, n_tools = response_text_only(round_data)
    redacted = f"\n[{n_tools} tool calls in the response were redacted]" if n_tools else ""
    searched = []
    for tc in round_data.get("toolCalls") or []:
        if tc.get("name") in SEARCH_TOOLS:
            args = (tc.get("arguments") or "").replace("\n", " ")
            searched.append(f"- {tc['name']} {args[:200]}")
    searched_block = "\n".join(searched) if searched else "- none"
    state = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"ASSISTANT RESPONSE (tool calls redacted):\n{response}{redacted}\n\n"
        f"MEMORY TOOL CALLS THIS ROUND MADE (arguments only):\n{searched_block}\n\n"
        f"PARENT ROUND EXISTS: {parent_exists}"
    )
    return state[:state_chars]


def walk_chain(rounds_dir: Path, target: str, max_depth: int = 0) -> list[dict]:
    """Links newest-first: target, parent, ..., root."""
    links, seen, rid = [], set(), target
    while rid and rid not in seen:
        if max_depth and len(links) >= max_depth:
            break
        seen.add(rid)
        data = load_round(rounds_dir, rid)
        if data is None:
            links.append({"id": rid, "missing": True})
            break
        parent_id = (data.get("parentId") or "").removesuffix(".json") or None
        links.append({"id": rid, "data": data, "parent_id": parent_id})
        rid = parent_id
    return links


def build_chain_context(rounds_dir: Path, target: str, chain_chars: int) -> str:
    """Parental prompt chain as evidence for the LLM stages (target excluded)."""
    lines = []
    for i, link in enumerate(walk_chain(rounds_dir, target)[1:], 1):  # skip target
        if link.get("missing"):
            lines.append(f"[parent {i}: {link['id']} — round file missing]")
            continue
        prompt = (link["data"].get("userPrompt") or "")[:1500]
        lines.append(f"[parent {i}] {link['id']}\nPROMPT: {prompt}")
    ctx = "\n\n".join(lines) if lines else "(no parent chain — this round has no parents)"
    return ctx[:chain_chars]


# ---- stages ----

def s0_filter(round_id: str, round_data: dict, parent_exists: str, api_key: str, args) -> float:
    """S0: jev round filter. Returns f in [0,1] (rank signal, never a gate)."""
    state = build_round_state(round_data, parent_exists, args.state_chars)
    fault_q = {
        "fault": {
            "type": "score",
            "instructions": (
                "Judge ONLY this round: the user prompt below and the "
                "assistant response to it. How likely is it that THIS round "
                "is faulty — that the assistant response mishandled this "
                "request compared to what a competent assistant should have "
                "done with the same input and the same context? Do not judge "
                "the user. 0 = defensible, 2 = cannot tell, 4 = clearly "
                "faulty."
            ),
            "criteria": [
                "0 — Defensible: a competent assistant would produce this response.",
                "1 — Mostly defensible: minor weaknesses only.",
                "2 — Cannot tell from the payload.",
                "3 — Probably faulty.",
                "4 — Clearly faulty.",
            ],
        }
    }
    answers = jev_request(state, api_key, fault_q)
    raw = float(answers["fault"].get("score", 2.0))
    return round(min(max(raw / 4.0, 0.0), 1.0), 4)


def s1_fault_type(round_id: str, round_data: dict, parent_exists: str, api_key: str, args) -> tuple[str, float]:
    """S1: jev provisional fault type via one noul question per taxonomy entry."""
    state = build_round_state(round_data, parent_exists, args.state_chars)
    answers = jev_request(state, api_key, FAULT_TYPE_QUESTIONS)
    probs = {t: float(answers[t].get("noul", 0.0)) for t in TAXONOMY}
    best = max(probs, key=probs.get)
    return best, probs[best]


def s2_root_cause(round_id: str, round_data: dict, rounds_dir: Path, api_key: str, args) -> dict:
    """S2: LLM root-cause analysis with the parental chain as context."""
    state = build_round_state(round_data, "see chain below", args.state_chars)
    chain = build_chain_context(rounds_dir, round_id, args.chain_chars)
    user = f"TARGET ROUND:\n{state}\n\nPARENTAL PROMPT CHAIN (oldest last):\n{chain}"
    out = parse_json_loose(llm_request(S2_SYSTEM, user, api_key, args.llm_model))
    return out


def s3_retype(round_id: str, round_data: dict, s2_out: dict, rounds_dir: Path, api_key: str, args) -> dict:
    state = build_round_state(round_data, "see chain below", args.state_chars)
    chain = build_chain_context(rounds_dir, round_id, args.chain_chars)
    user = (
        f"TARGET ROUND:\n{state}\n\nPARENTAL PROMPT CHAIN (oldest last):\n{chain}\n\n"
        f"FIRST ANALYSIS:\n{json.dumps(s2_out, indent=2)}"
    )
    return parse_json_loose(llm_request(S3_SYSTEM, user, api_key, args.llm_model))


def s4_revise(round_id: str, round_data: dict, s2_out: dict, s3_out: dict, api_key: str, args) -> dict:
    state = build_round_state(round_data, "see chain below", args.state_chars)
    user = (
        f"ROUND:\n{state}\n\nFIRST ANALYSIS:\n{json.dumps(s2_out, indent=2)}\n\n"
        f"CORRECTED FAULT TYPE:\n{json.dumps(s3_out, indent=2)}"
    )
    return parse_json_loose(llm_request(S4_SYSTEM, user, api_key, args.llm_model))


def bin_for(record: dict) -> str:
    s2 = record.get("s2") or {}
    s3 = record.get("s3") or {}
    if s3:
        ft = s3.get("fault_type")
        if ft == NO_FAULT:
            return NO_FAULT
        return ft or AMBIGUOUS
    v = s2.get("verdict")
    if v == "no_fault_within_round":
        return NO_FAULT
    if v == "external":
        return EXTERNAL
    if v == "ambiguous":
        return AMBIGUOUS
    if v == "fault_observed":
        return s2.get("fault_type") or AMBIGUOUS
    if v == "fault_kind_not_observed":
        return AMBIGUOUS  # only reached if S3 was skipped
    return AMBIGUOUS


def process_round(rid: str, rounds_dir: Path, api_key: str, args, stages: set) -> dict:
    record = {"id": rid, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data = load_round(rounds_dir, rid)
    if data is None:
        record["error"] = "round file missing"
        record["bin"] = AMBIGUOUS
        return record
    parent_id = (data.get("parentId") or "").removesuffix(".json") or None
    parent_exists = "yes" if parent_id else "no"
    record["prompt_preview"] = (data.get("userPrompt") or "")[:120]

    if "S0" in stages:
        f = s0_filter(rid, data, parent_exists, api_key, args)
        record["s0_f"] = f
        record["s0_filter_pass"] = True  # rank-based; filter never gates by default

    if "S1" in stages:
        ftype, p = s1_fault_type(rid, data, parent_exists, api_key, args)
        record["s1_fault_type"] = ftype
        record["s1_prob"] = p

    if "S2" in stages:
        s2_out = s2_root_cause(rid, data, rounds_dir, api_key, args)
        record["s2"] = s2_out
        if s2_out.get("verdict") == "fault_kind_not_observed" and "S3" in stages:
            s3_out = s3_retype(rid, data, s2_out, rounds_dir, api_key, args)
            record["s3"] = s3_out
            ft = s3_out.get("fault_type")
            if ft and ft != NO_FAULT and "S4" in stages:
                s4_out = s4_revise(rid, data, s2_out, s3_out, api_key, args)
                record["s4"] = s4_out
                record["s2"]["root_cause"] = s4_out.get("root_cause", s2_out.get("root_cause"))
                record["s2"]["evidence"] = s4_out.get("evidence", s2_out.get("evidence"))
                record["s2"]["fault_type"] = ft

    record["bin"] = bin_for(record)
    return record


# ---- eval set ----

def persist_eval_set(rounds_dir: Path, out_path: Path) -> None:
    """Freeze the evaluation set from chain-fault profiles + stage-2 survivors."""
    entries = []
    chain_path = Path("temp/jev-chain-fault.jsonl")
    if chain_path.exists():
        for line in chain_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            entries.append({
                "id": rec["target"],
                "source": "jev-chain-fault",
                "suspected_faulty": rec.get("top_origin"),
                "p_origin": rec.get("p_origin"),
            })
    s2_path = Path("temp/stage-2-jev-results.txt")
    s2_survivors = []
    if s2_path.exists():
        have = {e["id"] for e in entries}
        for line in s2_path.read_text().splitlines():
            rid = line.strip()
            if rid and rid not in have:
                entries.append({"id": rid, "source": "stage-2-jev-results", "suspected_faulty": None})
                s2_survivors.append(rid)
    # Add the parent round of each stage-2 survivor: if the survivor was a
    # genuine correction, the fault lives upstream — the parent is the
    # suspected faulty round we want in the eval set to measure recall.
    have = {e["id"] for e in entries}
    added = 0
    for rid in s2_survivors:
        f = rounds_dir / f"{rid}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        parent = (data.get("parentId") or "").removesuffix(".json") or None
        if not parent or parent in have or not (rounds_dir / f"{parent}.json").exists():
            continue
        entries.append({"id": parent, "source": "parent-of-stage-2-survivor", "child": rid, "suspected_faulty": None})
        have.add(parent)
        added += 1
    out_path.write_text(json.dumps({"version": 1, "entries": entries}, indent=2))
    print(f"persisted eval set: {len(entries)} entries ({added} parent-of-survivor entries added) -> {out_path}")


def load_eval_ids(eval_path: Path) -> list[str]:
    data = json.loads(eval_path.read_text())
    return [e["id"] for e in data["entries"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("round_ids", nargs="*")
    ap.add_argument("--ids-file")
    ap.add_argument("--rounds-dir", default=DEFAULT_ROUNDS_DIR)
    ap.add_argument("--top-k", type=int, default=50, help="S0 rank budget (max rounds carried past S0)")
    ap.add_argument("--top-frac", type=float, default=0.25, help="alternative: carry top fraction by f")
    ap.add_argument("--stages", default="S0,S1,S2,S3,S4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--state-chars", type=int, default=12000)
    ap.add_argument("--chain-chars", type=int, default=24000)
    ap.add_argument("--out", default="temp/fault-pipeline.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--persist-eval", action="store_true")
    ap.add_argument("--eval", action="store_true", help="run over the persisted eval set")
    ap.add_argument("--eval-set", default="temp/eval-set.json")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    if args.persist_eval:
        persist_eval_set(Path(args.rounds_dir), Path(args.eval_set))

    if args.eval:
        ids = load_eval_ids(Path(args.eval_set))
    elif args.ids_file:
        ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    elif args.round_ids:
        ids = args.round_ids
    else:
        ap.error("provide round ids, --ids-file, or --eval")
        return 1

    stages = {s.strip().upper() for s in args.stages.split(",") if s.strip()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resume: {len(done)} rounds already processed")
    todo = [r for r in ids if r not in done]
    print(f"pipeline: {len(todo)} of {len(ids)} rounds, stages={sorted(stages)}, "
          f"workers={args.workers}, llm={args.llm_model}")

    if args.dry_run:
        for rid in todo[: args.top_k]:
            data = load_round(Path(args.rounds_dir), rid)
            if data is None:
                print(f"DRY {rid}: round file missing")
                continue
            state = build_round_state(data, "yes", args.state_chars)
            print(f"DRY {rid}: state {len(state)} chars | {record_prompt_preview(data)}")
        return 0

    out_f = out_path.open("a")
    bins: dict[str, int] = {}
    errors = 0
    start = time.monotonic()

    def work(rid):
        return process_round(rid, Path(args.rounds_dir), api_key, args, stages)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rid = futures[fut]
            try:
                rec = fut.result()
                done[rid] = rec
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                bins[rec["bin"]] = bins.get(rec["bin"], 0) + 1
                print(f"[{i}/{len(todo)}] {rid} bin={rec['bin']} "
                      f"f={rec.get('s0_f', '-')} s1={rec.get('s1_fault_type', '-')} "
                      f"| {rec.get('prompt_preview', '')[:60]}")
            except Exception as exc:  # noqa: BLE001 — keep the pipeline alive
                errors += 1
                print(f"ERROR {rid}: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - start
    print(f"\ndone in {elapsed:.0f}s | bins: {json.dumps(bins, sort_keys=True)} | errors: {errors}")
    return 0


def record_prompt_preview(data: dict) -> str:
    return (data.get("userPrompt") or "")[:60].replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())
