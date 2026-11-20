"""CLI runner for paired multi-policy replay experiments."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers import make_controller
from src.data import build_sample_fdb_v2_dataset, load_traces
from src.engine import CalibratedMockEngine, VLLMEngine
from src.simulator import (
    DEFAULT_LOAD_RATES_RPS,
    ReplayHarness,
    RunSummary,
    TurnMetrics,
    generate_workload_schedule,
    save_summaries_json,
    save_turn_metrics_jsonl,
)

DEFAULT_POLICIES = [
    "no_draft",
    "fixed_interval",
]


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _format_table(summaries: Sequence[RunSummary]) -> str:
    grouped: Dict[tuple, List[RunSummary]] = defaultdict(list)
    for s in summaries:
        grouped[(s.load_regime, s.policy_name)].append(s)

    header = (
        f"{'Load':<13} {'Policy':<20} {'p50(ms)':>8} {'p95(ms)':>8} "
        f"{'SLO%':>7} {'Wait(ms)':>9} {'Drafted':>8} {'Accept':>8} {'Acc%':>7}"
    )
    lines = [header, "-" * len(header)]
    for load_regime in ("light", "intermediate", "heavy"):
        policies_in_load = [
            key for key in grouped.keys() if key[0] == load_regime
        ]
        # Preserve canonical policy order when present
        order_map = {name: i for i, name in enumerate(DEFAULT_POLICIES)}
        policies_in_load.sort(key=lambda k: order_map.get(k[1], 999))
        for key in policies_in_load:
            runs = grouped[key]
            p50 = float(np.mean([r.ttfs_p50_ms for r in runs]))
            p95 = float(np.mean([r.ttfs_p95_ms for r in runs]))
            slo = float(np.mean([r.slo_attainment_pct for r in runs]))
            wait = float(np.mean([r.mean_p0_queue_wait_ms for r in runs]))
            drafted = float(np.mean([r.total_drafted_tokens for r in runs]))
            accepted = float(np.mean([r.total_accepted_tokens for r in runs]))
            acc_pct = 100.0 * (accepted / drafted) if drafted > 0 else 0.0
            lines.append(
                f"{load_regime:<13} {key[1]:<20} {p50:>8.1f} {p95:>8.1f} "
                f"{slo:>6.1f}% {wait:>9.2f} {drafted:>8.0f} {accepted:>8.0f} "
                f"{acc_pct:>6.1f}%"
            )
        if policies_in_load:
            lines.append("-" * len(header))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared-GPU early-draft experiments.")
    parser.add_argument("--mode", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--model_name", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "fdb_v2_sample"))
    parser.add_argument("--split", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--loads", default="light,intermediate,heavy")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--target_turns", type=int, default=48)
    parser.add_argument("--warmup_turns", type=int, default=6)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--slo_target_ms", type=float, default=190.0)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trace_file = data_dir / f"{args.split}.jsonl"
    if not trace_file.exists():
        print(f"Generating sample FDB-v2 traces under {data_dir} ...")
        build_sample_fdb_v2_dataset(data_dir, seed=42)

    traces = load_traces(trace_file)
    print(f"Loaded {len(traces)} traces from {trace_file}")

    if args.mode == "mock":
        engine = CalibratedMockEngine()
    else:
        print(f"Loading VLLMEngine ({args.model_name}, dtype={args.dtype}) ...")
        engine = VLLMEngine(
            model_name=args.model_name,
            dtype=args.dtype,
        )

    engine.warmup()

    harness = ReplayHarness(
        engine=engine,
        slo_target_ms=args.slo_target_ms,
        max_batch_size=args.max_batch_size,
        warmup_turns=args.warmup_turns,
    )

    policies = _parse_csv(args.policies)
    loads = _parse_csv(args.loads)
    seeds = [int(s) for s in _parse_csv(args.seeds)]

    all_metrics: List[TurnMetrics] = []
    all_summaries: List[RunSummary] = []

    for load_name in loads:
        if load_name not in DEFAULT_LOAD_RATES_RPS:
            raise ValueError(f"Unknown load regime {load_name!r}")
        rps = DEFAULT_LOAD_RATES_RPS[load_name]
        for seed in seeds:
            schedule = generate_workload_schedule(
                traces=traces,
                arrival_rate_rps=rps,
                seed=seed,
                target_turns=args.target_turns,
            )
            for policy_name in policies:
                controller = make_controller(policy_name)
                metrics, summary = harness.run(
                    schedule=schedule,
                    controller=controller,
                    load_regime=load_name,
                    arrival_rate_rps=rps,
                    seed=seed,
                )
                all_metrics.extend(metrics)
                all_summaries.append(summary)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = out_dir / f"turns_{args.mode}_{args.split}.jsonl"
    summary_path = out_dir / f"summaries_{args.mode}_{args.split}.json"
    save_turn_metrics_jsonl(all_metrics, turns_path)
    save_summaries_json(all_summaries, summary_path)

    print("\n" + _format_table(all_summaries))
    print(f"\nSaved turn metrics to {turns_path}")
    print(f"Saved summaries to    {summary_path}")


if __name__ == "__main__":
    main()
