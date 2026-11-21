"""Audit saved paired runs without loading NumPy, vLLM, or a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import audit_turn_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turns", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on inconsistencies")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.turns.read_text().splitlines() if line.strip()]
    report = {"source": str(args.turns), **audit_turn_rows(rows)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Rows: {report['turn_rows']}; comparisons: {report['paired_comparisons']} "
          f"({report['token_comparisons']} token, {report['text_only_comparisons']} text-only)")
    for key in ("output_mismatches", "warmup_mismatches", "accounting_errors", "duplicate_turns", "missing_pairs"):
        print(f"{key}: {len(report[key])}")
    print("Direct greedy reference: not checked by this artifact audit")
    if args.strict and not report["artifact_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
