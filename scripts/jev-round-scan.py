#!/usr/bin/env python3
"""Scan semblr rounds with Jev (via OpenRouter Decisions API).

For each round in the rounds directory, asks Jev:
  1. How frustrated the user sounds in their prompt (score 0-4).
  2. Whether the round is a correction of a semblr round-discovery failure (noul).

Rounds with high frustration or a correction verdict are written to the
examination bin.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python3 scripts/jev-round-scan.py [--rounds-dir DIR] [--limit N]
      [--frustration-threshold 3.0] [--correction-threshold 0.7]
      [--workers 8] [--state-chars 12000] [--resume] [--dry-run]

Outputs:
  - temp/jev-scan-results.jsonl   (one record per round, all verdicts)
  - temp/examination-bin.txt      (round ids that tripped a threshold)

With --refine, re-examines flagged rounds from the examination bin:
  - builds a state from the flagged round's prompt + its parent round's
    response (tool calls redacted)
  - rounds with no parent are automatic false positives
  - asks Jev whether it is a genuine correction of a round-discovery
    failure or a benign reference to an old conversation

Refine outputs:
  - temp/refine-results.jsonl         (one record per refined round)
  - temp/false-positive.txt           (round ids judged benign / parentless)
  - temp/stage-2-jev-results.txt      (round ids that survived as corrections)
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
MODEL = "typesafe/jev-1.13"

QUESTIONS = {
    "frustration": {
        "type": "score",
        "instructions": (
            "How much frustration, irritation, or annoyance does the user "
            "express in this prompt? Consider complaints, exasperation, "
            "sarcasm, urgency, or corrections delivered with heat. A calm "
            "neutral request scores 0."
        ),
        "criteria": [
            "Calm and neutral",
            "Slightly impatient",
            "Clearly annoyed",
            "Frustrated or upset",
            "Angry or at the end of their rope",
        ],
    },
    "correction": {
        "type": "noul",
        "instructions": (
            "Is the user correcting the assistant's failure to find, recall, "
            "or surface a relevant past conversation round? Signals: the user "
            "says 'you already did this', 'we discussed this before', 'you "
            "forgot', 'search again', 'that's not what I asked', or points "
            "out that the assistant lost/misremembered context from earlier "
            "rounds. General corrections of code or answers do NOT count - "
            "only corrections about failed recall/discovery of past rounds."
        ),
    },
}


def build_state(round_data: dict, state_chars: int) -> str:
    prompt = round_data.get("userPrompt") or ""
    seq = round_data.get("responseSequence")
    if isinstance(seq, str):
        # Rounds store the response as a plain string; iterating it yields
        # single chars, which seg.get("text") silently drops -> empty response.
        response = seq
    else:
        response = ""
        for seg in seq or []:
            text = seg.get("text") if isinstance(seg, dict) else None
            if text:
                response += text + "\n"
    combined = f"USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}"
    return combined[:state_chars]


def query_jev(state: str, api_key: str, questions: dict | None = None) -> dict:
    payload = json.dumps(
        {"model": MODEL, "state": state, "questions": questions or QUESTIONS}
    ).encode()
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
    if not answers:
        # dump unknown shape for debugging
        raise ValueError(f"unexpected response shape: {list(body.keys())}")
    return answers


def classify(answers: dict, frustration_threshold: float, correction_threshold: float) -> dict:
    fr = answers.get("frustration", {})
    co = answers.get("correction", {})
    frustration = fr.get("score", 0.0)
    confidence = fr.get("confidence", 0.0)
    correction_p = co.get("noul", 0.0)
    return {
        "frustration": frustration,
        "frustration_confidence": confidence,
        "correction_probability": correction_p,
        "flagged": (
            frustration >= frustration_threshold
            or correction_p >= correction_threshold
        ),
    }


REFINE_QUESTIONS = {
    "genuine_correction": {
        "type": "noul",
        "instructions": (
            "You are given a user prompt and the response of the round that "
            "immediately preceded it (tool calls redacted). Decide: is the "
            "user genuinely correcting the assistant's failure to find, "
            "recall, or use the right prior conversation round? A genuine "
            "correction means the assistant missed or misremembered context "
            "it should have had. By contrast, a benign reference is when the "
            "user intentionally recalls an OLD conversation from long ago "
            "('remember that thing we discussed seven weeks ago?') — that is "
            "a deliberate query, not a failure of the assistant. Ordinary "
            "follow-up work that correctly builds on the previous round is "
            "also not a correction."
        ),
    },
}


def build_refine_state(round_data: dict, parent_data: dict, state_chars: int) -> str:
    """State = flagged round's prompt + parent's redacted response."""
    prompt = round_data.get("userPrompt") or ""
    texts = []
    n_tools = 0
    for seg in parent_data.get("responseSegments") or []:
        if seg.get("type") == "toolCall":
            n_tools += 1
        elif seg.get("type") == "text" and seg.get("text"):
            texts.append(seg["text"])
    redacted = "\n".join(texts)
    if n_tools:
        redacted += (
            f"\n\n[{n_tools} tool calls in the parent response were redacted]"
        )
    combined = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"PREVIOUS ROUND RESPONSE (tool calls redacted):\n{redacted}"
    )
    return combined[:state_chars]


def refine(args, api_key: str) -> int:
    rounds_dir = Path(args.rounds_dir)
    in_path = Path("temp/examination-bin.txt")
    out_path = Path("temp/refine-results.jsonl")
    fp_path = Path("temp/false-positive.txt")
    s2_path = Path("temp/stage-2-jev-results.txt")
    if not in_path.exists():
        print(f"ERROR: {in_path} not found - run the scan first", file=sys.stderr)
        return 1
    ids = [l.strip() for l in in_path.read_text().splitlines() if l.strip()]

    done = {}
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} rounds already refined")
    todo = [r for r in ids if r not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"refining {len(todo)} of {len(ids)} flagged rounds with {args.workers} workers")

    def work(rid: str):
        rpath = rounds_dir / f"{rid}.json"
        if not rpath.exists():
            return {"id": rid, "outcome": "false_positive", "reason": "round file missing"}
        data = json.loads(rpath.read_text())
        parent_id = (data.get("parentId") or "").removesuffix(".json")
        if not parent_id:
            return {"id": rid, "outcome": "false_positive", "reason": "no parent round"}
        ppath = rounds_dir / f"{parent_id}.json"
        if not ppath.exists():
            return {"id": rid, "outcome": "false_positive", "reason": "parent file missing"}
        parent = json.loads(ppath.read_text())
        state = build_refine_state(data, parent, args.state_chars)
        answers = query_jev(state, api_key, REFINE_QUESTIONS)
        p = answers["genuine_correction"].get("noul", 0.0)
        outcome = "correction" if p >= args.correction_threshold else "false_positive"
        return {
            "id": rid,
            "parent": parent_id,
            "outcome": outcome,
            "correction_probability": p,
            "prompt_preview": (data.get("userPrompt") or "")[:120],
        }

    out_f = out_path.open("a")
    counts = {"correction": 0, "false_positive": 0}
    errors = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rid = futures[fut]
            try:
                rec = fut.result()
                done[rid] = rec
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                if rec["outcome"] in counts:
                    counts[rec["outcome"]] += 1
                    print(f"{rec['outcome'].upper()} {rid} "
                          f"p={rec.get('correction_probability', '-')} "
                          f"| {rec.get('prompt_preview', '')[:60]}")
            except (error.HTTPError, error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors += 1
                print(f"ERROR {rid}: {exc}", file=sys.stderr)
            if i % 50 == 0:
                rate = i / (time.monotonic() - start)
                print(f"... {i}/{len(todo)} ({rate:.1f}/s, {errors} errors)")
    out_f.close()

    # Rewrite the two output files from ALL refined records (idempotent).
    fp_ids = [r["id"] for r in done.values() if r["outcome"] == "false_positive"]
    s2_ids = [r["id"] for r in done.values() if r["outcome"] == "correction"]
    fp_path.write_text("\n".join(sorted(fp_ids)) + "\n")
    s2_path.write_text("\n".join(sorted(s2_ids)) + "\n")

    print(
        f"\ndone: {len(todo)} refined, {counts['correction']} corrections -> {s2_path}, "
        f"{counts['false_positive']} false positives -> {fp_path}, {errors} errors"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds-dir", default=os.path.expanduser("~/.pi/agent/semblr/rounds"))
    ap.add_argument("--limit", type=int, default=0, help="max rounds to scan (0 = all)")
    ap.add_argument("--frustration-threshold", type=float, default=3.0)
    ap.add_argument("--correction-threshold", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--state-chars", type=int, default=12000)
    ap.add_argument("--resume", action="store_true", help="skip rounds already in results")
    ap.add_argument("--dry-run", action="store_true", help="print state for one round and exit")
    ap.add_argument("--refine", action="store_true", help="re-examine flagged rounds from the examination bin")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 1

    rounds_dir = Path(args.rounds_dir)
    round_files = sorted(rounds_dir.glob("*.json"))
    if args.limit:
        round_files = round_files[: args.limit]
    if args.refine:
        return refine(args, api_key)

    if not round_files:
        print(f"ERROR: no rounds found in {rounds_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        data = json.loads(round_files[0].read_text())
        print(f"round: {data['id']}")
        print(build_state(data, args.state_chars))
        return 0

    out_path = Path("temp/jev-scan-results.jsonl")
    bin_path = Path("temp/examination-bin.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} rounds already scanned")

    todo = [p for p in round_files if p.stem not in done]
    print(f"scanning {len(todo)} of {len(round_files)} rounds with {args.workers} workers")

    out_f = out_path.open("a")
    flagged = []
    errors = 0
    start = time.monotonic()

    def scan(path: Path):
        data = json.loads(path.read_text())
        state = build_state(data, args.state_chars)
        answers = query_jev(state, api_key)
        verdict = classify(answers, args.frustration_threshold, args.correction_threshold)
        return {
            "id": data["id"],
            "prompt_preview": (data.get("userPrompt") or "")[:120],
            "session": data.get("sessionLabel"),
            **verdict,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            path = futures[fut]
            try:
                record = fut.result()
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                if record["flagged"]:
                    flagged.append(record)
                    print(
                        f"FLAG {record['id']} frustration={record['frustration']:.2f} "
                        f"correction={record['correction_probability']:.2f} "
                        f"| {record['prompt_preview'][:60]}"
                    )
            except (error.HTTPError, error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors += 1
                print(f"ERROR {path.stem}: {exc}", file=sys.stderr)
            if i % 100 == 0:
                rate = i / (time.monotonic() - start)
                print(f"... {i}/{len(todo)} ({rate:.1f}/s, {errors} errors)")

    out_f.close()

    # Append flagged ids to the examination bin (deduped, preserving prior entries)
    existing = set()
    if bin_path.exists():
        existing = set(bin_path.read_text().split())
    new_ids = [r["id"] for r in flagged if r["id"] not in existing]
    with bin_path.open("a") as bf:
        for rid in new_ids:
            bf.write(rid + "\n")

    print(
        f"\ndone: {len(todo)} scanned, {len(flagged)} flagged "
        f"({len(new_ids)} new in {bin_path}), {errors} errors"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
