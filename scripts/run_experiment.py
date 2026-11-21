"""CLI runner for paired multi-policy replay experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    save_timeline_jsonl,
    save_turn_metrics_jsonl,
)
from src.validation import audit_turn_rows

DEFAULT_POLICIES = [
    "no_draft",
    "fixed_interval",
    "load_reuse_adaptive",
    "reuse_only",
]


def _capture_command(command: List[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, cwd=PROJECT_ROOT)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def _manifest(args: argparse.Namespace, trace_file: Path) -> Dict[str, Any]:
    versions = {}
    for package in ("numpy", "vllm", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = sorted(list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py")))
    sources.extend([PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "arguments": vars(args),
        "git_commit": _capture_command(["git", "rev-parse", "HEAD"]),
        "git_status": _capture_command(["git", "status", "--short"]),
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "dataset_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "gpu": _capture_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) if args.mode == "vllm" else None,
        "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "cache_policy": "reset before every policy/load/seed run",
        "warmup_policy": "first W final arrivals in offered schedule; identical turn identities across policies",
        "timing": "discrete-event replay; sums engine call durations, excludes controller/formatting/logging overhead",
        "ttfs_definition": "first .?! after >=4 tokens, excluding selected abbreviations; fallback at 24 tokens or response end",
        "backend": "synthetic roofline" if args.mode == "mock" else "completed one-token vLLM requests with prefix caching; not persistent decode requests",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0},
    }


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
    parser.add_argument("--model_revision", default=None, help="Pin a model/tokenizer checkpoint revision")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "fdb_v2_sample"))
    parser.add_argument("--split", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--loads", default="light,intermediate,heavy")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--target_turns", type=int, default=48)
    parser.add_argument("--warmup_turns", type=int, default=6)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--slo_target_ms", type=float, default=190.0)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--skip_output_validation", action="store_true",
                        help="Diagnostic only: skip direct greedy checks; mark the run unvalidated")
    parser.add_argument(
        "--no_timeline",
        action="store_true",
        help="Disable step-level GPU timeline JSONL output.",
    )
    args = parser.parse_args()

    if args.warmup_turns < 0 or args.target_turns <= args.warmup_turns:
        parser.error("target_turns must exceed warmup_turns >= 0")
    if args.max_batch_size < 1 or args.max_new_tokens < 1:
        parser.error("max_batch_size and max_new_tokens must be positive")
    policies = _parse_csv(args.policies)
    loads = _parse_csv(args.loads)
    seeds = [int(s) for s in _parse_csv(args.seeds)]
    if not policies or not loads or not seeds:
        parser.error("policies, loads, and seeds must be nonempty")
    canonical_names = [make_controller(name).name for name in policies]
    if len(set(canonical_names)) != len(policies) or len(set(loads)) != len(loads) or len(set(seeds)) != len(seeds):
        parser.error("policies (including aliases), loads, and seeds must be unique")
    if set(loads) - set(DEFAULT_LOAD_RATES_RPS):
        parser.error("Unknown load regime")

    data_dir = Path(args.data_dir)
    trace_file = data_dir / f"{args.split}.jsonl"
    if not trace_file.exists():
        print(f"Generating sample FDB-v2 traces under {data_dir} ...")
        build_sample_fdb_v2_dataset(data_dir, seed=42)

    traces = load_traces(trace_file)
    if not traces:
        parser.error("The trace file contains no turns")
    print(f"Loaded {len(traces)} traces from {trace_file}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.mode}_{args.split}"
    turns_path = out_dir / f"turns_{suffix}.jsonl"
    summary_path = out_dir / f"summaries_{suffix}.json"
    timeline_path = out_dir / f"timeline_{suffix}.jsonl"
    manifest_path = out_dir / f"manifest_{suffix}.json"
    validation_path = out_dir / f"validation_{suffix}.json"
    if manifest_path.exists() or turns_path.exists() or summary_path.exists() or timeline_path.exists() or validation_path.exists():
        parser.error("Output artifacts already exist; choose a fresh output_dir to preserve run provenance")
    manifest = _manifest(args, trace_file)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.mode == "mock":
        engine = CalibratedMockEngine()
    else:
        print(f"Loading VLLMEngine ({args.model_name}, dtype={args.dtype}) ...")
        engine = VLLMEngine(
            model_name=args.model_name,
            dtype=args.dtype,
            revision=args.model_revision,
        )

    engine.warmup()

    harness = ReplayHarness(
        engine=engine,
        slo_target_ms=args.slo_target_ms,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        warmup_turns=args.warmup_turns,
        record_timeline=not args.no_timeline,
    )

    all_metrics: List[TurnMetrics] = []
    all_summaries: List[RunSummary] = []
    all_timeline: List[Dict[str, Any]] = []
    prompt_lookup = {}
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

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
            for ev in schedule:
                prompt_lookup[(load_name, seed, ev.stream_conversation_id, ev.trace.turn_id)] = (ev.trace.history, ev.trace.final.text)
            for policy_name in policies:
                controller = make_controller(policy_name)
                engine.reset_prefix_cache()
                metrics, summary = harness.run(
                    schedule=schedule,
                    controller=controller,
                    load_regime=load_name,
                    arrival_rate_rps=rps,
                    seed=seed,
                )
                if len(metrics) != len(schedule):
                    raise RuntimeError("Replay did not complete every scheduled turn")
                all_metrics.extend(metrics)
                all_summaries.append(summary)
                if harness.last_timeline:
                    all_timeline.extend(harness.last_timeline)
                # Preserve completed runs even if later validation or GPU work fails.
                save_turn_metrics_jsonl(all_metrics, turns_path)
                save_summaries_json(all_summaries, summary_path)
                if all_timeline:
                    save_timeline_jsonl(all_timeline, timeline_path)

    audit = audit_turn_rows([m.to_dict() for m in all_metrics])
    reference_mismatches = []
    if not args.skip_output_validation:
        # Checks run after all timed work so they do not warm any measured caches.
        engine.reset_prefix_cache()
        reference_cache = {}
        for metric in all_metrics:
            history, text = prompt_lookup[(metric.load_regime, metric.seed, metric.conversation_id, metric.turn_id)]
            prompt_key = json.dumps([history, text], sort_keys=True)
            if prompt_key not in reference_cache:
                reference_cache[prompt_key] = engine.generate_greedy_reference(history, text, args.max_new_tokens)
            reference = reference_cache[prompt_key]
            if metric.response_tokens != reference:
                reference_mismatches.append({"load": metric.load_regime, "seed": metric.seed,
                    "policy": metric.policy_name, "conversation_id": metric.conversation_id,
                    "turn_id": metric.turn_id, "response_tokens": metric.response_tokens,
                    "reference_tokens": reference})
        audit["greedy_reference_status"] = "failed" if reference_mismatches else "passed"
    audit["reference_checked_turns"] = 0 if args.skip_output_validation else len(all_metrics)
    audit["reference_mismatches"] = reference_mismatches
    validation_path.write_text(json.dumps(audit, indent=2) + "\n")
    failed = not audit["artifact_consistent"] or bool(reference_mismatches)
    manifest["status"] = "invalid_outputs" if failed else ("unvalidated" if args.skip_output_validation else "validated")
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["validation_file"] = validation_path.name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nMeans of per-seed run summaries (not pooled percentiles):")
    print(_format_table(all_summaries))
    print(f"\nSaved turn metrics to {turns_path}")
    print(f"Saved summaries to    {summary_path}")
    if all_timeline:
        print(f"Saved timeline to     {timeline_path}")
    print(f"Output checks: {manifest['status']} ({validation_path})")
    if failed:
        raise SystemExit("Output checks failed; artifacts retained for diagnosis, not a valid equivalent-output comparison.")


if __name__ == "__main__":
    main()
