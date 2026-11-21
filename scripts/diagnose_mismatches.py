"""Show where replayed outputs first diverge from the direct greedy reference.

Usage (from the repo root, on a PACE login node):
    uv run python scripts/diagnose_mismatches.py results/noise_floor/run-<jobid>-bfloat16

Reads validation_*.json in that folder. For every reference mismatch it reports
the first token position where the two sequences differ. Late divergence
(after many tokens) suggests floating-point near-ties; divergence at position 0
or 1 suggests a real difference in how the two code paths build the prompt.

If the model tokenizer is in the local Hugging Face cache, it also decodes the
text around the divergence point so you can read what each path produced.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def first_divergence(a, b):
    """Index of the first differing token, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model_name", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--no_decode", action="store_true", help="Skip loading the tokenizer")
    args = parser.parse_args()

    files = sorted(args.run_dir.glob("validation_*.json"))
    if not files:
        raise SystemExit(f"No validation_*.json in {args.run_dir}")
    validation = json.loads(files[0].read_text())
    mismatches = validation.get("reference_mismatches", [])
    print(f"File: {files[0]}")
    print(f"Turns checked: {validation.get('reference_checked_turns')}   "
          f"Mismatches: {len(mismatches)}\n")
    if not mismatches:
        return

    tokenizer = None
    if not args.no_decode:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as exc:  # tokenizer is optional
            print(f"(Could not load tokenizer, showing token IDs only: {exc})\n")

    def show(tokens):
        if tokenizer is None:
            return str(tokens)
        return repr(tokenizer.decode(tokens, skip_special_tokens=False))

    positions = Counter()
    kinds = Counter()
    for m in mismatches:
        got, ref = m["response_tokens"], m["reference_tokens"]
        pos = first_divergence(got, ref)
        if pos is None:
            kind = "length only (one output is a prefix of the other)"
            pos = min(len(got), len(ref))
        else:
            kind = "token differs"
        kinds[kind] += 1
        positions[pos] += 1

        print(f"[{m['load']}] {m['conversation_id']} turn {m['turn_id']} ({m['policy']})")
        print(f"  lengths: replay={len(got)} reference={len(ref)}   "
              f"first difference at token {pos} ({kind})")
        lo = max(0, pos - 8)
        print(f"  shared prefix ...{show(got[lo:pos])}")
        print(f"  replay    -> {show(got[pos:pos + 8])}")
        print(f"  reference -> {show(ref[pos:pos + 8])}\n")

    print("Summary")
    print("  kind of mismatch:", dict(kinds))
    print("  first-difference position -> count:", dict(sorted(positions.items())))
    early = sum(c for p, c in positions.items() if p <= 1)
    print(f"  diverging at token 0 or 1: {early} of {len(mismatches)}")
    if early == len(mismatches):
        print("  => Every mismatch starts immediately: look for a prompt/formatting difference.")
    elif early == 0:
        print("  => Mismatches start mid-response: consistent with floating-point near-ties.")
    else:
        print("  => Mixed: inspect the early cases first.")


if __name__ == "__main__":
    main()
