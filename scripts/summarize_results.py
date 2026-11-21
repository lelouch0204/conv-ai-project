"""Aggregate per-seed latency summaries without pooling turn percentiles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Sequence


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["load_regime"], row["policy_name"])].append(row)
    result = []
    load_order = {"light": 0, "intermediate": 1, "heavy": 2}
    policy_order = {name: i for i, name in enumerate(
        ("no_draft", "fixed_interval", "load_reuse_adaptive", "reuse_only"))}
    for (load, policy), runs in sorted(grouped.items(), key=lambda item: (
            load_order.get(item[0][0], 99), item[0][0],
            policy_order.get(item[0][1], 99), item[0][1])):
        seeds = sorted(r["seed"] for r in runs)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed in {load}/{policy}: {seeds}")
        if len({r["arrival_rate_rps"] for r in runs}) != 1:
            raise ValueError(f"Mixed arrival rates in {load}/{policy}")
        if len({r["slo_target_ms"] for r in runs}) != 1:
            raise ValueError(f"Mixed SLO targets in {load}/{policy}")
        out = {"load_regime": load, "policy_name": policy, "seeds": seeds,
               "num_runs": len(runs), "evaluated_turns": sum(r["num_turns"] for r in runs)}
        for metric in ("ttfs_p50_ms", "ttfs_p95_ms", "slo_attainment_pct", "mean_p0_queue_wait_ms"):
            values = [float(r[metric]) for r in runs]
            out[metric] = {"mean": round(statistics.mean(values), 3),
                           "sample_sd": round(statistics.stdev(values), 3) if len(values) > 1 else None}
        drafted = sum(r["total_drafted_tokens"] for r in runs)
        accepted = sum(r["total_accepted_tokens"] for r in runs)
        out["overall_draft_acceptance_rate"] = accepted / drafted if drafted else 0.0
        result.append(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.summaries.read_text())
    validation_path = args.summaries.with_name(args.summaries.name.replace("summaries_", "validation_", 1))
    status = "missing"
    if validation_path != args.summaries and validation_path.exists():
        v = json.loads(validation_path.read_text())
        status = "passed" if v.get("greedy_reference_status") == "passed" and v.get("artifact_consistent") else "failed_or_unvalidated"
    report = {"source": str(args.summaries), "output_validation": status,
              "aggregation": "Mean and sample SD of per-seed run quantiles; not pooled turn quantiles. SD is null for one run.",
              "groups": aggregate(rows)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Output validation: {status}; mean +/- sample SD across workload seeds")
    print(f"{'Load':<14}{'Policy':<23}{'Runs':>5}{'Turns':>7}{'p50 ms':>21}{'p95 ms':>21}")
    for g in report["groups"]:
        def fmt(metric):
            m = g[metric]
            return f"{m['mean']:.1f} +/- {m['sample_sd']:.1f}" if m['sample_sd'] is not None else f"{m['mean']:.1f} (one run)"
        print(f"{g['load_regime']:<14}{g['policy_name']:<23}{g['num_runs']:>5}{g['evaluated_turns']:>7}"
              f"{fmt('ttfs_p50_ms'):>21}{fmt('ttfs_p95_ms'):>21}")


if __name__ == "__main__":
    main()
