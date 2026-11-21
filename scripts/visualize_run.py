"""Generate an interactive HTML simulation replay (and optional ASCII view) from experiment outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import heapq
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in pure stdlib environments
    np = None  # type: ignore[assignment]


_ASCII_SPARK = " _.,:-=+*#%@"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _load_summaries(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def _swap_prefix(path: Path, old_prefix: str, new_prefix: str, new_ext: str) -> Path:
    stem = path.stem
    if stem.startswith(old_prefix):
        suffix = stem[len(old_prefix) :]
        return path.with_name(f"{new_prefix}{suffix}{new_ext}")
    return path.with_name(f"{new_prefix}_{stem}{new_ext}")


def resolve_input_paths(
    input_path: Path,
    summaries_arg: Optional[str] = None,
    timeline_arg: Optional[str] = None,
    output_arg: Optional[str] = None,
) -> Tuple[Path, Optional[Path], Optional[Path], Path]:
    """Resolve ``(turns_path, summaries_path, timeline_path, output_html_path)``."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input path does not exist: {src}")

    summaries_path: Optional[Path] = Path(summaries_arg) if summaries_arg else None
    timeline_path: Optional[Path] = Path(timeline_arg) if timeline_arg else None

    if src.suffix == ".json":
        if summaries_path is None:
            summaries_path = src
        if src.stem.startswith("summaries_"):
            candidate_turns = _swap_prefix(src, "summaries_", "turns_", ".jsonl")
        else:
            candidate_turns = src.with_name("turns.jsonl")
        turns_path = candidate_turns
        suffix_label = src.stem[len("summaries_") :] if src.stem.startswith("summaries_") else src.stem
    else:
        turns_path = src
        suffix_label = src.stem[len("turns_") :] if src.stem.startswith("turns_") else src.stem
        if summaries_path is None:
            cand_sum = _swap_prefix(src, "turns_", "summaries_", ".json")
            if cand_sum.exists():
                summaries_path = cand_sum

    if not turns_path.exists():
        raise FileNotFoundError(f"Could not find turn metrics JSONL file: {turns_path}")

    if timeline_path is None:
        cand_tl = turns_path.with_name(f"timeline_{suffix_label}.jsonl")
        if not cand_tl.exists():
            cand_tl = _swap_prefix(turns_path, "turns_", "timeline_", ".jsonl")
        if cand_tl.exists():
            timeline_path = cand_tl
    elif not timeline_path.exists():
        raise FileNotFoundError(f"Specified timeline file does not exist: {timeline_path}")

    if output_arg:
        out_path = Path(output_arg)
    else:
        out_name = f"replay_{suffix_label}.html"
        out_path = turns_path.with_name(out_name)

    return turns_path, summaries_path, timeline_path, out_path


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not valid:
        return []
    merged: List[Tuple[float, float]] = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_overlap_sum(
    disjoint_intervals: Sequence[Tuple[float, float]],
    b0: float,
    b1: float,
) -> float:
    total = 0.0
    for start, end in disjoint_intervals:
        if end <= b0:
            continue
        if start >= b1:
            break
        total += min(b1, end) - max(b0, start)
    return total


def _peak_concurrency_in_window(
    intervals: Sequence[Tuple[float, float]],
    b0: float,
    b1: float,
) -> int:
    clipped: List[Tuple[float, float]] = []
    for start, end in intervals:
        cs = max(b0, start)
        ce = min(b1, end)
        if ce > cs:
            clipped.append((cs, ce))
    if not clipped:
        return 0
    clipped.sort()
    active_ends: List[float] = []
    peak = 0
    for cs, ce in clipped:
        while active_ends and active_ends[0] <= cs + 1e-9:
            heapq.heappop(active_ends)
        heapq.heappush(active_ends, ce)
        if len(active_ends) > peak:
            peak = len(active_ends)
    return peak


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.percentile(np.asarray(values, dtype=float), pct))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _user_sort_key(conv_id: str) -> Tuple[int, str]:
    if "#s" in conv_id:
        prefix, tail = conv_id.rsplit("#s", 1)
        if tail.isdigit():
            return (int(tail), prefix)
    return (10**9, conv_id)


def _extract_draft_bursts(
    steps: Sequence[Dict[str, Any]],
    merge_gap_ms: float = 25.0,
) -> Dict[Tuple[str, int], List[Tuple[float, float]]]:
    raw_by_turn: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
    for st in steps:
        t0 = float(st.get("t0", 0.0))
        dt = float(st.get("dt", 0.0))
        if dt <= 0.0:
            continue
        t1 = t0 + dt
        for label in st.get("c1", []) or []:
            if ":t" not in label:
                continue
            conv_id, turn_str = label.rsplit(":t", 1)
            try:
                turn_id = int(turn_str)
            except ValueError:
                continue
            raw_by_turn[(conv_id, turn_id)].append((t0, t1))

    merged_by_turn: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for key, intervals in raw_by_turn.items():
        intervals.sort()
        bursts: List[Tuple[float, float]] = [intervals[0]]
        for s, e in intervals[1:]:
            ps, pe = bursts[-1]
            if s - pe <= merge_gap_ms:
                bursts[-1] = (ps, max(pe, e))
            else:
                bursts.append((s, e))
        merged_by_turn[key] = [(round(a, 2), round(b, 2)) for a, b in bursts]
    return merged_by_turn


def _build_turn_record(
    raw: Dict[str, Any],
    bursts_map: Dict[Tuple[str, int], List[Tuple[float, float]]],
) -> Dict[str, Any]:
    conv_id = str(raw["conversation_id"])
    turn_id = int(raw["turn_id"])
    u_start = float(raw["utterance_start_ms"])
    u_end = max(u_start, float(raw["utterance_end_ms"]))
    comp = max(u_end, float(raw["completion_ms"]))

    wait_ms = max(0.0, float(raw.get("p0_queue_wait_ms", 0.0)))
    ttft_ms = max(0.0, float(raw.get("ttft_ms", 0.0)))
    ttfs_ms = max(0.0, float(raw.get("ttfs_ms", 0.0)))
    total_ms = max(0.0, float(raw.get("total_response_ms", comp - u_end)))

    q_end = max(u_end, min(comp, u_end + wait_ms))
    ttft_end = max(q_end, min(comp, u_end + ttft_ms))
    ttfs_end = max(ttft_end, min(comp, u_end + ttfs_ms))

    bursts = bursts_map.get((conv_id, turn_id), [])
    return {
        "conversation_id": conv_id,
        "turn_id": turn_id,
        "key": f"{conv_id}:t{turn_id}",
        "family": str(raw.get("family", "")),
        "pacing": str(raw.get("pacing", "")),
        "synthetic_revision": bool(raw.get("synthetic_revision", False)),
        "is_warmup": bool(raw.get("is_warmup", False)),
        "timed_out": bool(raw.get("timed_out", False)),
        "utterance_start_ms": round(u_start, 2),
        "utterance_end_ms": round(u_end, 2),
        "q_end_ms": round(q_end, 2),
        "ttft_end_ms": round(ttft_end, 2),
        "ttfs_end_ms": round(ttfs_end, 2),
        "completion_ms": round(comp, 2),
        "p0_queue_wait_ms": round(wait_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "ttfs_ms": round(ttfs_ms, 2),
        "total_response_ms": round(total_ms, 2),
        "drafted_tokens": int(raw.get("drafted_tokens", 0)),
        "accepted_tokens": int(raw.get("accepted_tokens", 0)),
        "canceled_tokens": int(raw.get("canceled_tokens", 0)),
        "acceptance_rate": round(float(raw.get("acceptance_rate", 0.0)), 4),
        "response_text": str(raw.get("response_text", "")),
        "draft_bursts": [[a, b] for a, b in bursts],
    }


def _compute_bins_from_turns(
    turns: Sequence[Dict[str, Any]],
    bin_ms: float,
    t_max_ms: float,
) -> List[Dict[str, Any]]:
    speak_intervals = [
        (t["utterance_start_ms"], t["utterance_end_ms"])
        for t in turns
        if t["utterance_end_ms"] > t["utterance_start_ms"]
    ]
    queue_intervals = [
        (t["utterance_end_ms"], t["q_end_ms"])
        for t in turns
        if t["q_end_ms"] > t["utterance_end_ms"] + 1e-6
    ]
    p0_gpu_intervals = [
        (t["q_end_ms"], t["completion_ms"])
        for t in turns
        if t["completion_ms"] > t["q_end_ms"] + 1e-6
    ]
    merged_p0 = _merge_intervals(p0_gpu_intervals)

    n_bins = max(1, int(math.ceil(t_max_ms / bin_ms)))
    bins: List[Dict[str, Any]] = []
    for idx in range(n_bins):
        b0 = idx * bin_ms
        b1 = min(t_max_ms, (idx + 1) * bin_ms)
        dur = max(1e-6, b1 - b0)
        p0_busy_ms = _interval_overlap_sum(merged_p0, b0, b1)
        p0_pct = min(100.0, round(100.0 * p0_busy_ms / dur, 2))
        bins.append(
            {
                "t0": round(b0, 2),
                "t1": round(b1, 2),
                "p0_util_pct": p0_pct,
                "p1_util_pct": 0.0,
                "total_gpu_busy_pct": p0_pct,
                "active_speakers": _peak_concurrency_in_window(speak_intervals, b0, b1),
                "max_batch_p0": _peak_concurrency_in_window(p0_gpu_intervals, b0, b1),
                "max_batch_p1": 0,
                "max_queue_p0": _peak_concurrency_in_window(queue_intervals, b0, b1),
                "max_queue_p1": 0,
            }
        )
    return bins


def _compute_bins_from_timeline(
    turns: Sequence[Dict[str, Any]],
    steps: Sequence[Dict[str, Any]],
    bin_ms: float,
    t_max_ms: float,
) -> List[Dict[str, Any]]:
    speak_intervals = [
        (t["utterance_start_ms"], t["utterance_end_ms"])
        for t in turns
        if t["utterance_end_ms"] > t["utterance_start_ms"]
    ]
    n_bins = max(1, int(math.ceil(t_max_ms / bin_ms)))
    p0_ms_bins = [0.0] * n_bins
    p1_ms_bins = [0.0] * n_bins
    max_b0 = [0] * n_bins
    max_b1 = [0] * n_bins
    max_q0 = [0] * n_bins
    max_q1 = [0] * n_bins

    for st in steps:
        s0 = float(st.get("t0", 0.0))
        dt = float(st.get("dt", 0.0))
        if dt <= 0.0:
            continue
        s1 = s0 + dt
        p0 = int(st.get("p0", 0))
        p1 = int(st.get("p1", 0))
        q0 = int(st.get("q0", 0))
        q1 = int(st.get("q1", 0))
        if p0 + p1 <= 0:
            continue

        start_idx = max(0, int(math.floor(s0 / bin_ms)))
        end_idx = min(n_bins - 1, int(math.floor(max(s0, s1 - 1e-9) / bin_ms)))
        w0 = p0 / (p0 + p1)
        w1 = p1 / (p0 + p1)

        for idx in range(start_idx, end_idx + 1):
            b0 = idx * bin_ms
            b1 = min(t_max_ms, (idx + 1) * bin_ms)
            ov = min(b1, s1) - max(b0, s0)
            if ov <= 0.0:
                continue
            p0_ms_bins[idx] += ov * w0
            p1_ms_bins[idx] += ov * w1
            if p0 > max_b0[idx]:
                max_b0[idx] = p0
            if p1 > max_b1[idx]:
                max_b1[idx] = p1
            if q0 > max_q0[idx]:
                max_q0[idx] = q0
            if q1 > max_q1[idx]:
                max_q1[idx] = q1

    bins: List[Dict[str, Any]] = []
    for idx in range(n_bins):
        b0 = idx * bin_ms
        b1 = min(t_max_ms, (idx + 1) * bin_ms)
        dur = max(1e-6, b1 - b0)
        p0_pct = 100.0 * p0_ms_bins[idx] / dur
        p1_pct = 100.0 * p1_ms_bins[idx] / dur
        total_pct = p0_pct + p1_pct
        if total_pct > 100.0:
            scale = 100.0 / total_pct
            p0_pct *= scale
            p1_pct *= scale
            total_pct = 100.0
        bins.append(
            {
                "t0": round(b0, 2),
                "t1": round(b1, 2),
                "p0_util_pct": round(p0_pct, 2),
                "p1_util_pct": round(p1_pct, 2),
                "total_gpu_busy_pct": round(total_pct, 2),
                "active_speakers": _peak_concurrency_in_window(speak_intervals, b0, b1),
                "max_batch_p0": max_b0[idx],
                "max_batch_p1": max_b1[idx],
                "max_queue_p0": max_q0[idx],
                "max_queue_p1": max_q1[idx],
            }
        )
    return bins


def _derive_summary(
    turns: Sequence[Dict[str, Any]],
    policy_name: str,
    load_regime: str,
    seed: int,
    slo_target_ms: float,
) -> Dict[str, Any]:
    steady = [t for t in turns if not t["is_warmup"]]
    eval_turns = steady
    warmup_count = sum(1 for t in turns if t["is_warmup"])
    if not eval_turns:
        return {
            "policy_name": policy_name,
            "load_regime": load_regime,
            "arrival_rate_rps": 0.0,
            "seed": seed,
            "num_turns": 0,
            "warmup_turns": warmup_count,
            "ttfs_p50_ms": 0.0,
            "ttfs_p95_ms": 0.0,
            "ttfs_mean_ms": 0.0,
            "ttft_p50_ms": 0.0,
            "ttft_p95_ms": 0.0,
            "slo_target_ms": slo_target_ms,
            "slo_attainment_pct": 0.0,
            "total_drafted_tokens": 0,
            "total_accepted_tokens": 0,
            "total_canceled_tokens": 0,
            "mean_acceptance_rate": 0.0,
            "mean_p0_queue_wait_ms": 0.0,
            "timeout_count": 0,
        }

    ttfs_vals = [float(t["ttfs_ms"]) for t in eval_turns]
    ttft_vals = [float(t["ttft_ms"]) for t in eval_turns]
    wait_vals = [float(t["p0_queue_wait_ms"]) for t in eval_turns]
    drafted = sum(int(t["drafted_tokens"]) for t in eval_turns)
    accepted = sum(int(t["accepted_tokens"]) for t in eval_turns)
    canceled = sum(int(t["canceled_tokens"]) for t in eval_turns)
    slo_hits = sum(1 for t in eval_turns if not t["timed_out"] and t["ttfs_ms"] <= slo_target_ms)

    return {
        "policy_name": policy_name,
        "load_regime": load_regime,
        "arrival_rate_rps": 0.0,
        "seed": seed,
        "num_turns": len(eval_turns),
        "warmup_turns": warmup_count,
        "ttfs_p50_ms": round(_percentile(ttfs_vals, 50.0), 2),
        "ttfs_p95_ms": round(_percentile(ttfs_vals, 95.0), 2),
        "ttfs_mean_ms": round(sum(ttfs_vals) / len(ttfs_vals), 2),
        "ttft_p50_ms": round(_percentile(ttft_vals, 50.0), 2),
        "ttft_p95_ms": round(_percentile(ttft_vals, 95.0), 2),
        "slo_target_ms": slo_target_ms,
        "slo_attainment_pct": round(100.0 * slo_hits / len(eval_turns), 2),
        "total_drafted_tokens": drafted,
        "total_accepted_tokens": accepted,
        "total_canceled_tokens": canceled,
        "mean_acceptance_rate": round((accepted / drafted) if drafted > 0 else 0.0, 4),
        "mean_p0_queue_wait_ms": round(sum(wait_vals) / len(wait_vals), 2),
        "timeout_count": sum(1 for t in eval_turns if t["timed_out"]),
    }


def _compute_run_aggregates(bins: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not bins:
        return {
            "avg_gpu_util_pct": 0.0,
            "avg_p0_util_pct": 0.0,
            "avg_p1_util_pct": 0.0,
            "peak_gpu_util_pct": 0.0,
            "peak_speakers": 0,
            "peak_batch": 0,
            "peak_queue": 0,
        }
    total_dur = 0.0
    p0_area = 0.0
    p1_area = 0.0
    peak_gpu = 0.0
    peak_spk = 0
    peak_batch = 0
    peak_q = 0
    for b in bins:
        dur = max(1e-6, float(b["t1"]) - float(b["t0"]))
        total_dur += dur
        p0_area += float(b["p0_util_pct"]) * dur
        p1_area += float(b["p1_util_pct"]) * dur
        peak_gpu = max(peak_gpu, float(b["total_gpu_busy_pct"]))
        peak_spk = max(peak_spk, int(b["active_speakers"]))
        peak_batch = max(peak_batch, int(b["max_batch_p0"]) + int(b["max_batch_p1"]))
        peak_q = max(peak_q, int(b["max_queue_p0"]) + int(b["max_queue_p1"]))
    avg_p0 = p0_area / total_dur
    avg_p1 = p1_area / total_dur
    return {
        "avg_gpu_util_pct": round(avg_p0 + avg_p1, 2),
        "avg_p0_util_pct": round(avg_p0, 2),
        "avg_p1_util_pct": round(avg_p1, 2),
        "peak_gpu_util_pct": round(peak_gpu, 2),
        "peak_speakers": peak_spk,
        "peak_batch": peak_batch,
        "peak_queue": peak_q,
    }


def build_visualization_payload(
    turns_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]] = (),
    timeline_rows: Sequence[Dict[str, Any]] = (),
    bin_ms: Optional[float] = None,
    max_batch_size: int = 8,
    slo_target_ms: float = 190.0,
    source_name: str = "",
) -> Dict[str, Any]:
    """Compile turn metrics, optional summaries, and optional step traces into replay JSON."""
    turns_by_key: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    order_keys: List[Tuple[str, int, str]] = []
    for row in turns_rows:
        key = (str(row.get("load_regime", "custom")), int(row.get("seed", 1)), str(row.get("policy_name", "unknown")))
        if key not in turns_by_key:
            order_keys.append(key)
        turns_by_key[key].append(row)

    sum_by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in summary_rows:
        key = (str(row.get("load_regime", "custom")), int(row.get("seed", 1)), str(row.get("policy_name", "unknown")))
        sum_by_key[key] = row

    tl_by_key: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in timeline_rows:
        key = (str(row.get("load_regime", "custom")), int(row.get("seed", 1)), str(row.get("policy_name", "unknown")))
        tl_by_key[key].append(row)

    # Align t_max_ms across all policies sharing (load_regime, seed)
    group_tmax: Dict[Tuple[str, int], float] = defaultdict(lambda: 100.0)
    group_users: Dict[Tuple[str, int], set] = defaultdict(set)
    for (load_regime, seed, policy_name), rows in turns_by_key.items():
        gkey = (load_regime, seed)
        for r in rows:
            group_tmax[gkey] = max(group_tmax[gkey], float(r.get("completion_ms", 0.0)))
            group_users[gkey].add(str(r["conversation_id"]))
        for st in tl_by_key.get((load_regime, seed, policy_name), []):
            group_tmax[gkey] = max(group_tmax[gkey], float(st.get("t0", 0.0)) + float(st.get("dt", 0.0)))

    groups_dict: Dict[Tuple[str, int], Dict[str, Any]] = {}
    groups_order: List[Tuple[str, int]] = []

    for key in order_keys:
        load_regime, seed, policy_name = key
        gkey = (load_regime, seed)
        raw_turns = turns_by_key[key]
        steps = tl_by_key.get(key, [])
        has_timeline = len(steps) > 0
        t_max_ms = max(100.0, round(group_tmax[gkey] * 1.01, 2))

        eff_bin_ms = float(bin_ms) if bin_ms is not None else min(1000.0, max(100.0, round(t_max_ms / 200.0, 1)))
        bursts_map = _extract_draft_bursts(steps) if has_timeline else {}

        turns = [_build_turn_record(r, bursts_map) for r in raw_turns]
        turns.sort(key=lambda t: (t["utterance_start_ms"], t["utterance_end_ms"], t["key"]))

        users = sorted(
            group_users[gkey] | {t["conversation_id"] for t in turns},
            key=_user_sort_key,
        )

        if has_timeline:
            bins = _compute_bins_from_timeline(turns, steps, eff_bin_ms, t_max_ms)
            step_caps = [int(st.get("cap", max_batch_size)) for st in steps if st.get("cap")]
            run_cap = max(step_caps) if step_caps else max_batch_size
        else:
            bins = _compute_bins_from_turns(turns, eff_bin_ms, t_max_ms)
            run_cap = max_batch_size

        raw_sum = sum_by_key.get(key)
        run_slo = float(raw_sum.get("slo_target_ms", slo_target_ms)) if raw_sum else float(slo_target_ms)
        summary = dict(raw_sum) if raw_sum else _derive_summary(turns, policy_name, load_regime, seed, run_slo)
        agg = _compute_run_aggregates(bins)

        if gkey not in groups_dict:
            groups_order.append(gkey)
            groups_dict[gkey] = {
                "group_id": f"{load_regime}|seed={seed}",
                "load_regime": load_regime,
                "seed": seed,
                "arrival_rate_rps": float(summary.get("arrival_rate_rps", 0.0)),
                "t_max_ms": t_max_ms,
                "bin_ms": eff_bin_ms,
                "slo_target_ms": run_slo,
                "max_batch_size": run_cap,
                "users": users,
                "policies": {},
                "policy_order": [],
            }

        grp = groups_dict[gkey]
        if summary.get("arrival_rate_rps") and not grp["arrival_rate_rps"]:
            grp["arrival_rate_rps"] = float(summary["arrival_rate_rps"])
        if policy_name not in grp["policies"]:
            grp["policy_order"].append(policy_name)
        grp["policies"][policy_name] = {
            "policy_name": policy_name,
            "has_timeline": has_timeline,
            "summary": summary,
            "aggregates": agg,
            "turns": turns,
            "bins": bins,
        }

    return {
        "version": 1,
        "source_name": source_name,
        "groups": [groups_dict[gkey] for gkey in groups_order],
    }


def _ascii_lane_for_user(
    user_turns: Sequence[Dict[str, Any]],
    t_max_ms: float,
    width: int = 72,
    has_timeline: bool = False,
) -> str:
    chars = ["."] * width
    priority = {".": 0, "=": 1, "d": 2, "q": 3, "#": 4}

    def paint(start_ms: float, end_ms: float, ch: str) -> None:
        if end_ms <= start_ms or t_max_ms <= 0:
            return
        i0 = max(0, min(width - 1, int(math.floor(start_ms / t_max_ms * width))))
        i1 = max(i0, min(width - 1, int(math.floor(max(start_ms, end_ms - 1e-6) / t_max_ms * width))))
        p = priority[ch]
        for idx in range(i0, i1 + 1):
            if p > priority[chars[idx]]:
                chars[idx] = ch

    for t in user_turns:
        paint(t["utterance_start_ms"], t["utterance_end_ms"], "=")
        if has_timeline:
            for bs, be in t.get("draft_bursts", []):
                paint(bs, be, "d")
        if t["q_end_ms"] > t["utterance_end_ms"] + 0.1:
            paint(t["utterance_end_ms"], t["q_end_ms"], "q")
        paint(t["q_end_ms"], t["completion_ms"], "#")

    return "".join(chars)


def format_ascii_report(payload: Dict[str, Any], width: int = 72) -> str:
    """Format a compact terminal summary, GPU sparkline, and user Gantt timeline."""
    lines: List[str] = []
    for grp in payload.get("groups", []):
        t_max_ms = float(grp["t_max_ms"])
        slo_ms = float(grp["slo_target_ms"])
        rps = grp.get("arrival_rate_rps", 0.0)
        rps_str = f" | {rps} rps" if rps else ""

        for policy_name in grp["policy_order"]:
            run = grp["policies"][policy_name]
            s = run["summary"]
            agg = run["aggregates"]
            mode_tag = "Full Timeline (P0+P1)" if run["has_timeline"] else "Turn-Only (P0 envelope estimate)"

            lines.append("=" * 88)
            lines.append(
                f"REPLAY: {grp['load_regime']} (seed={grp['seed']}{rps_str}) | "
                f"Policy: {policy_name} | Mode: {mode_tag}"
            )
            lines.append(
                f"TTFS p50={s['ttfs_p50_ms']}ms  p95={s['ttfs_p95_ms']}ms  mean={s['ttfs_mean_ms']}ms | "
                f"SLO(<={slo_ms:.0f}ms)={s['slo_attainment_pct']}% | "
                f"AvgBusy={agg['avg_gpu_util_pct']}% (P0={agg['avg_p0_util_pct']}%, P1={agg['avg_p1_util_pct']}%) | "
                f"Acc={s['total_accepted_tokens']}/{s['total_drafted_tokens']} (Can={s['total_canceled_tokens']})"
            )
            lines.append(
                "Legend: '.' idle  '=' speaking  'd' P1 draft  'q' P0 queued  '#' P0 GPU active"
            )
            lines.append("-" * 88)

            by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for t in run["turns"]:
                by_user[t["conversation_id"]].append(t)

            for uid in grp["users"]:
                short_uid = uid if len(uid) <= 22 else f"...{uid[-19:]}"
                lane = _ascii_lane_for_user(
                    by_user.get(uid, []),
                    t_max_ms=t_max_ms,
                    width=width,
                    has_timeline=run["has_timeline"],
                )
                lines.append(f"{short_uid:<22} |{lane}|")

            # GPU utilization sparkline quantized into width buckets
            bins = run["bins"]
            spark_chars: List[str] = []
            for col in range(width):
                b_start = (col / width) * t_max_ms
                b_end = ((col + 1) / width) * t_max_ms
                peak_u = 0.0
                for b in bins:
                    if b["t1"] > b_start and b["t0"] < b_end:
                        peak_u = max(peak_u, float(b["total_gpu_busy_pct"]))
                level = min(len(_ASCII_SPARK) - 1, int(round(peak_u / 100.0 * (len(_ASCII_SPARK) - 1))))
                spark_chars.append(_ASCII_SPARK[level])
            lines.append(f"{'Busy proxy (peak/col)':<22} |{''.join(spark_chars)}|")
            lines.append(f"{'Time (0 -> ' + f'{t_max_ms/1000:.1f}s)':<22}  {'0s':<{width - 6}}{t_max_ms/1000:>5.1f}s")
            lines.append("")
            lines.append(
                f"{'User':<24} {'Turn':>4} {'Family':<15} {'Wait':>7} {'TTFT':>7} "
                f"{'TTFS':>7} {'Total':>8} {'Acc/Drf':>9} {'Flags'}"
            )
            for t in run["turns"]:
                flags: List[str] = []
                if t["is_warmup"]:
                    flags.append("warmup")
                if t["synthetic_revision"]:
                    flags.append("rev")
                if t["ttfs_ms"] > slo_ms:
                    flags.append(">SLO")
                short_u = t["conversation_id"] if len(t["conversation_id"]) <= 24 else f"...{t['conversation_id'][-21:]}"
                lines.append(
                    f"{short_u:<24} {t['turn_id']:>4} {t['family']:<15} "
                    f"{t['p0_queue_wait_ms']:>6.1f}m {t['ttft_ms']:>6.1f}m "
                    f"{t['ttfs_ms']:>6.1f}m {t['total_response_ms']:>7.1f}m "
                    f"{t['accepted_tokens']:>4}/{t['drafted_tokens']:<4} {','.join(flags) or '-'}"
                )
    return "\n".join(lines)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shared-GPU Voice Draft Replay — __TITLE__</title>
<style>
:root {
  --bg: #0b101b;
  --panel: #131b2b;
  --panel2: #192338;
  --border: #26344f;
  --text: #e6edf7;
  --muted: #93a4bf;
  --speak: #3b82f6;
  --draft: #a855f7;
  --queue: #f59e0b;
  --ttft: #06b6d4;
  --ttfs: #10b981;
  --bad: #ef4444;
  --tail: #059669;
  --accent: #38bdf8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 14px 18px 28px;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px;
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: 0.2px; }
.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--panel2);
  color: var(--muted);
}
.badge.full { color: #c4b5fd; border-color: #6d28d9; background: rgba(109, 40, 217, 0.16); }
.badge.turn { color: #fcd34d; border-color: #b45309; background: rgba(180, 83, 9, 0.16); }

.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 12px;
  margin-bottom: 10px;
}
.ctrl-group { display: flex; align-items: center; gap: 6px; }
label { color: var(--muted); font-size: 12px; }
select, button, input[type="range"] {
  background: var(--panel2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 8px;
  font-size: 12px;
}
button { cursor: pointer; font-weight: 500; }
button:hover { border-color: var(--accent); color: #fff; }
button.primary { background: #1d4ed8; border-color: #3b82f6; color: #fff; }
.scrubber { flex: 1 1 240px; min-width: 180px; padding: 0; }

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
  gap: 8px;
  margin-bottom: 10px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 11px;
}
.card-title { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }
.card-val { font-size: 17px; font-weight: 700; margin-top: 3px; }
.card-sub { color: var(--muted); font-size: 11px; margin-top: 3px; line-height: 1.35; }
.pos { color: #34d399; }
.neg { color: #f87171; }

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px 6px;
  margin-bottom: 10px;
  position: relative;
}
.panel-hdr {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.panel-title { font-weight: 600; font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--muted); }
.leg-item { display: inline-flex; align-items: center; gap: 4px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

canvas { width: 100%; display: block; cursor: crosshair; }
#tooltip {
  position: fixed;
  pointer-events: none;
  display: none;
  z-index: 999;
  background: rgba(11, 16, 27, 0.96);
  border: 1px solid #3b82f6;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 11.5px;
  line-height: 1.45;
  max-width: 360px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}

.bottom-grid {
  display: grid;
  grid-template-columns: minmax(320px, 1.05fr) minmax(380px, 1.35fr);
  gap: 10px;
}
@media (max-width: 1024px) {
  .bottom-grid { grid-template-columns: 1fr; }
}
.inspector-body { font-size: 12px; line-height: 1.45; }
.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
  gap: 6px;
  margin: 8px 0;
}
.kv { background: var(--panel2); padding: 6px 8px; border-radius: 5px; border: 1px solid var(--border); }
.kv .k { font-size: 10.5px; color: var(--muted); }
.kv .v { font-size: 13px; font-weight: 600; margin-top: 2px; }
.resp-box {
  background: #0d1422;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 8px;
  white-space: pre-wrap;
  max-height: 140px;
  overflow: auto;
  font-size: 12px;
  color: #dbeafe;
}
.table-wrap { max-height: 360px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--border); }
th { position: sticky; top: 0; background: var(--panel2); color: var(--muted); font-weight: 600; z-index: 2; }
tr { cursor: pointer; }
tr:hover { background: rgba(56, 189, 248, 0.08); }
tr.selected { background: rgba(59, 130, 246, 0.22); }
.pill {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
}
.pill.ok { background: rgba(16, 185, 129, 0.18); color: #34d399; }
.pill.miss { background: rgba(239, 68, 68, 0.2); color: #f87171; }
.pill.warm { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Shared-GPU Voice Draft Simulation Replay <span id="hdr-source" class="muted mono" style="font-size:12px;color:#93a4bf;"></span></h1>
  </div>
  <div id="mode-badge" class="badge"></div>
</header>

<div class="controls">
  <div class="ctrl-group">
    <label for="sel-group">Workload</label>
    <select id="sel-group"></select>
  </div>
  <div class="ctrl-group">
    <label for="sel-policy">Policy</label>
    <select id="sel-policy"></select>
  </div>
  <div class="ctrl-group">
    <label for="sel-compare">Compare</label>
    <select id="sel-compare"></select>
  </div>
  <div class="ctrl-group">
    <button id="btn-play" class="primary">Play</button>
    <button id="btn-reset">Reset</button>
    <button id="btn-prev" title="Previous turn ([)">&lt; Turn</button>
    <button id="btn-next" title="Next turn (])">Turn &gt;</button>
  </div>
  <div class="ctrl-group">
    <label for="sel-speed">Speed</label>
    <select id="sel-speed">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1">1x</option>
      <option value="2" selected>2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label for="sel-zoom">Zoom</label>
    <select id="sel-zoom">
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
      <option value="8">8x</option>
      <option value="16">16x</option>
    </select>
    <label><input type="checkbox" id="chk-follow" checked> Follow</label>
  </div>
  <input id="rng-time" class="scrubber" type="range" min="0" max="1000" value="0" step="1">
  <span id="lbl-clock" class="mono" style="min-width:138px;text-align:right;">0.00 / 0.00 s</span>
</div>

<div class="cards" id="cards"></div>

<div class="panel">
  <div class="panel-hdr">
    <span class="panel-title">Users &amp; Turns Timeline <span style="color:var(--muted);font-weight:400;font-size:11.5px;">(click turn to inspect, drag background to scrub, Shift+drag when zoomed to pan)</span></span>
    <div class="legend">
      <span class="leg-item"><span class="swatch" style="background:var(--speak)"></span>Speaking</span>
      <span class="leg-item"><span class="swatch" style="background:var(--draft)"></span>P1 Draft</span>
      <span class="leg-item"><span class="swatch" style="background:var(--queue)"></span>P0 Queue</span>
      <span class="leg-item"><span class="swatch" style="background:var(--ttft)"></span>Prefill/TTFT</span>
      <span class="leg-item"><span class="swatch" style="background:var(--ttfs)"></span>TTFS (&le;SLO)</span>
      <span class="leg-item"><span class="swatch" style="background:var(--bad)"></span>TTFS (&gt;SLO)</span>
      <span class="leg-item"><span class="swatch" style="background:var(--tail)"></span>Tail Decode</span>
    </div>
  </div>
  <canvas id="cv-lanes"></canvas>
</div>

<div class="panel">
  <div class="panel-hdr">
    <span class="panel-title">Scheduled Occupancy &amp; Concurrency Over Time</span>
    <div class="legend">
      <span class="leg-item"><span class="swatch" style="background:rgba(16,185,129,0.55)"></span>P0 GPU %</span>
      <span class="leg-item"><span class="swatch" style="background:rgba(168,85,247,0.55)"></span>P1 Draft GPU %</span>
      <span class="leg-item"><span class="swatch" style="background:#3b82f6"></span>Active Speakers</span>
      <span class="leg-item"><span class="swatch" style="background:#22d3ee"></span>GPU Batch</span>
      <span class="leg-item"><span class="swatch" style="background:#f59e0b"></span>P0 Queue</span>
    </div>
  </div>
  <canvas id="cv-gpu"></canvas>
</div>

<div class="panel">
  <div class="panel-hdr">
    <span class="panel-title">Turn Latency Over Time (at utterance end)</span>
    <div class="legend">
      <span class="leg-item"><span class="swatch" style="background:#10b981"></span>TTFS (&le;SLO)</span>
      <span class="leg-item"><span class="swatch" style="background:#ef4444"></span>TTFS (&gt;SLO)</span>
      <span class="leg-item"><span class="swatch" style="background:#06b6d4"></span>TTFT</span>
      <span class="leg-item"><span class="swatch" style="background:#f59e0b"></span>Queue Wait</span>
      <span class="leg-item"><span class="swatch" style="background:#c084fc"></span>Compare TTFS (dashed)</span>
    </div>
  </div>
  <canvas id="cv-lat"></canvas>
</div>

<div class="bottom-grid">
  <div class="panel">
    <div class="panel-hdr">
      <span class="panel-title">Turn Inspector</span>
      <span id="insp-sub" class="mono" style="color:var(--muted);font-size:11px;">Click any turn bar or table row</span>
    </div>
    <div id="inspector" class="inspector-body" style="color:var(--muted);">No turn selected yet.</div>
  </div>

  <div class="panel">
    <div class="panel-hdr">
      <span class="panel-title">All Turns</span>
      <span id="tbl-sub" class="mono" style="color:var(--muted);font-size:11px;"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>User</th>
            <th>T</th>
            <th>Family</th>
            <th>End(s)</th>
            <th>Wait</th>
            <th>TTFT</th>
            <th>TTFS</th>
            <th>Total</th>
            <th>Acc/Drf</th>
            <th id="th-delta">Delta</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script id="replay-data" type="application/json">__PAYLOAD_JSON__</script>
<script>
(function() {
  const DATA = JSON.parse(document.getElementById("replay-data").textContent);
  const GUTTER = 190;
  const RPAD = 20;

  const state = {
    groupIdx: 0,
    policy: "",
    compare: "",
    tMs: 0,
    playing: false,
    speed: 2,
    zoom: 1,
    viewStartMs: 0,
    selectedKey: null,
    lastTs: null,
    hoverTurn: null,
  };

  const els = {
    source: document.getElementById("hdr-source"),
    badge: document.getElementById("mode-badge"),
    selGroup: document.getElementById("sel-group"),
    selPolicy: document.getElementById("sel-policy"),
    selCompare: document.getElementById("sel-compare"),
    btnPlay: document.getElementById("btn-play"),
    btnReset: document.getElementById("btn-reset"),
    btnPrev: document.getElementById("btn-prev"),
    btnNext: document.getElementById("btn-next"),
    selSpeed: document.getElementById("sel-speed"),
    selZoom: document.getElementById("sel-zoom"),
    chkFollow: document.getElementById("chk-follow"),
    rngTime: document.getElementById("rng-time"),
    lblClock: document.getElementById("lbl-clock"),
    cards: document.getElementById("cards"),
    cvLanes: document.getElementById("cv-lanes"),
    cvGpu: document.getElementById("cv-gpu"),
    cvLat: document.getElementById("cv-lat"),
    inspector: document.getElementById("inspector"),
    inspSub: document.getElementById("insp-sub"),
    tblBody: document.getElementById("tbl-body"),
    tblSub: document.getElementById("tbl-sub"),
    thDelta: document.getElementById("th-delta"),
    tooltip: document.getElementById("tooltip"),
  };

  if (DATA.source_name) els.source.textContent = "(" + DATA.source_name + ")";

  function curGroup() { return DATA.groups[state.groupIdx]; }
  function curRun() { const g = curGroup(); return g.policies[state.policy]; }
  function cmpRun() { const g = curGroup(); return state.compare && g.policies[state.compare] ? g.policies[state.compare] : null; }

  function pct(arr, p) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const pos = (p / 100) * (s.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return lo === hi ? s[lo] : s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo);
  }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function shortUser(u) {
    const idx = u.indexOf("#s");
    if (idx >= 0) return u.slice(idx) + " · " + u.slice(0, idx).replace(/^dev_|^held_out_/, "");
    return u;
  }

  function populateGroups() {
    els.selGroup.innerHTML = "";
    DATA.groups.forEach((g, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      const rps = g.arrival_rate_rps ? ` · ${g.arrival_rate_rps} rps` : "";
      opt.textContent = `${g.load_regime} · seed ${g.seed}${rps}`;
      els.selGroup.appendChild(opt);
    });
    state.groupIdx = 0;
    populatePolicies();
  }

  function populatePolicies() {
    const g = curGroup();
    els.selPolicy.innerHTML = "";
    els.selCompare.innerHTML = '<option value="">None</option>';
    g.policy_order.forEach(p => {
      const o1 = document.createElement("option");
      o1.value = p; o1.textContent = p;
      els.selPolicy.appendChild(o1);
      const o2 = document.createElement("option");
      o2.value = p; o2.textContent = p;
      els.selCompare.appendChild(o2);
    });
    state.policy = g.policy_order[0] || "";
    state.compare = g.policy_order.length > 1 ? g.policy_order[1] : "";
    els.selPolicy.value = state.policy;
    els.selCompare.value = state.compare;
    state.tMs = Math.min(g.t_max_ms, 0);
    state.viewStartMs = 0;
    state.selectedKey = (curRun() && curRun().turns[0]) ? curRun().turns[0].key : null;
    syncScrubberRange();
    updateAll();
  }

  function syncScrubberRange() {
    const g = curGroup();
    els.rngTime.max = String(Math.ceil(g.t_max_ms));
    els.rngTime.value = String(Math.round(state.tMs));
  }

  function viewWindow() {
    const g = curGroup();
    const span = g.t_max_ms / state.zoom;
    let s = state.viewStartMs;
    if (els.chkFollow.checked && state.zoom > 1) {
      if (state.tMs < s + span * 0.1 || state.tMs > s + span * 0.9) {
        s = state.tMs - span * 0.25;
      }
    }
    s = Math.max(0, Math.min(g.t_max_ms - span, s));
    state.viewStartMs = s;
    return [s, s + span];
  }

  function xFromTime(t, plotW, v0, v1) {
    return GUTTER + ((t - v0) / Math.max(1e-6, v1 - v0)) * plotW;
  }
  function timeFromX(x, plotW, v0, v1) {
    return v0 + ((x - GUTTER) / Math.max(1, plotW)) * (v1 - v0);
  }

  function setupCanvas(cv, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = cv.clientWidth || 900;
    cv.style.height = cssH + "px";
    if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
      cv.width = Math.round(cssW * dpr);
      cv.height = Math.round(cssH * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: cssW, h: cssH, plotW: Math.max(100, cssW - GUTTER - RPAD) };
  }

  function niceTicks(v0, v1, maxTicks) {
    const span = Math.max(1, v1 - v0);
    const raw = span / maxTicks;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / pow;
    const step = norm < 1.5 ? pow : norm < 3.5 ? 2 * pow : norm < 7.5 ? 5 * pow : 10 * pow;
    const start = Math.ceil(v0 / step) * step;
    const ticks = [];
    for (let t = start; t <= v1 + 1e-6; t += step) ticks.push(t);
    return ticks;
  }

  function userStatusAt(uid, turns, tMs, hasTimeline) {
    let seenDone = false;
    for (const tr of turns) {
      if (tr.conversation_id !== uid) continue;
      if (tMs >= tr.utterance_start_ms && tMs < tr.utterance_end_ms) {
        if (hasTimeline && (tr.draft_bursts || []).some(b => tMs >= b[0] && tMs <= b[1])) {
          return ["DRAFTING", "#c084fc", "rgba(168,85,247,0.22)"];
        }
        return ["SPEAKING", "#60a5fa", "rgba(59,130,246,0.2)"];
      }
      if (tMs >= tr.utterance_end_ms && tMs < tr.q_end_ms) {
        return ["QUEUED", "#fbbf24", "rgba(245,158,11,0.2)"];
      }
      if (tMs >= tr.q_end_ms && tMs < tr.ttft_end_ms) {
        return ["PREFILL", "#22d3ee", "rgba(6,182,212,0.2)"];
      }
      if (tMs >= tr.ttft_end_ms && tMs <= tr.completion_ms) {
        return ["DECODING", "#34d399", "rgba(16,185,129,0.2)"];
      }
      if (tMs > tr.completion_ms) seenDone = true;
    }
    return seenDone ? ["DONE", "#6ee7b7", "rgba(16,185,129,0.09)"] : ["IDLE", "#64748b", "rgba(100,116,139,0.12)"];
  }

  function findBinAt(run, tMs) {
    if (!run || !run.bins.length) return null;
    for (const b of run.bins) {
      if (tMs >= b.t0 && tMs <= b.t1) return b;
    }
    return run.bins[run.bins.length - 1];
  }

  function renderCards() {
    const g = curGroup();
    const run = curRun();
    const cmp = cmpRun();
    const b = findBinAt(run, state.tMs);
    const tMs = state.tMs;

    let speaking = 0, onGpu = 0, queued = 0, doneCount = 0;
    const completed = [];
    for (const tr of run.turns) {
      if (tMs >= tr.utterance_start_ms && tMs < tr.utterance_end_ms) speaking++;
      if (tMs >= tr.utterance_end_ms && tMs < tr.q_end_ms) queued++;
      if (tMs >= tr.q_end_ms && tMs <= tr.completion_ms) onGpu++;
      if (tMs >= tr.completion_ms) { doneCount++; completed.push(tr); }
    }
    const steadyDone = completed.filter(x => !x.is_warmup);
    const evalDone = steadyDone.length ? steadyDone : completed;
    const evalLabel = steadyDone.length ? `steady (${steadyDone.length})` : (completed.length ? `warmup (${completed.length})` : "waiting");

    const ttfsArr = evalDone.map(x => x.ttfs_ms);
    const p50 = pct(ttfsArr, 50), p95 = pct(ttfsArr, 95);
    const sloHits = evalDone.filter(x => !x.timed_out && x.ttfs_ms <= g.slo_target_ms).length;
    const sloPct = evalDone.length ? (100 * sloHits / evalDone.length) : 0;

    let cmpLine = "";
    if (cmp) {
      const cDone = cmp.turns.filter(x => tMs >= x.completion_ms);
      const cSteady = cDone.filter(x => !x.is_warmup);
      const cEval = cSteady.length ? cSteady : cDone;
      if (cEval.length && evalDone.length) {
        const dp50 = p50 - pct(cEval.map(x => x.ttfs_ms), 50);
        const cls = dp50 <= 0 ? "pos" : "neg";
        const sign = dp50 > 0 ? "+" : "";
        cmpLine = ` · vs ${cmp.policy_name}: <span class="${cls}">${sign}${dp50.toFixed(1)} ms</span>`;
      } else {
        cmpLine = ` · vs ${cmp.policy_name}: pending`;
      }
    }

    const drf = completed.reduce((a, x) => a + x.drafted_tokens, 0);
    const acc = completed.reduce((a, x) => a + x.accepted_tokens, 0);
    const can = completed.reduce((a, x) => a + x.canceled_tokens, 0);
    const accRate = drf > 0 ? (100 * acc / drf).toFixed(1) + "%" : "0.0%";

    const instGpu = b ? b.total_gpu_busy_pct.toFixed(1) : "0.0";
    const instP0 = b ? b.p0_util_pct.toFixed(1) : "0.0";
    const instP1 = b ? b.p1_util_pct.toFixed(1) : "0.0";
    const batchNow = b ? (b.max_batch_p0 + b.max_batch_p1) : 0;

    els.cards.innerHTML = `
      <div class="card">
        <div class="card-title">Simulation Clock</div>
        <div class="card-val mono">${(tMs / 1000).toFixed(2)}s <span style="font-size:12px;color:var(--muted);">(${(100 * tMs / g.t_max_ms).toFixed(0)}%)</span></div>
        <div class="card-sub">${esc(g.load_regime)} · seed ${g.seed} · cap ${g.max_batch_size}</div>
      </div>
      <div class="card">
        <div class="card-title">Active Users at t</div>
        <div class="card-val mono">${speaking} speak · ${onGpu} GPU · ${queued} wait</div>
        <div class="card-sub">${doneCount} / ${run.turns.length} turns completed · ${g.users.length} streams</div>
      </div>
      <div class="card">
        <div class="card-title">Scheduled Occupancy Proxy</div>
        <div class="card-val mono">${instGpu}% <span style="font-size:12px;color:var(--muted);">now</span> · ${run.aggregates.avg_gpu_util_pct.toFixed(1)}% <span style="font-size:12px;color:var(--muted);">avg</span></div>
        <div class="card-sub">Bin P0=${instP0}% P1=${instP1}% · peak batch ${batchNow}/${g.max_batch_size}</div>
      </div>
      <div class="card">
        <div class="card-title">Running TTFS (${evalLabel})</div>
        <div class="card-val mono">${evalDone.length ? p50.toFixed(1) + " ms" : "—"} <span style="font-size:12px;color:var(--muted);">p50</span></div>
        <div class="card-sub">p95 ${evalDone.length ? p95.toFixed(1) + " ms" : "—"} · SLO ${evalDone.length ? sloPct.toFixed(0) + "%" : "—"}${cmpLine}</div>
      </div>
      <div class="card">
        <div class="card-title">Speculation Tokens (&le; t)</div>
        <div class="card-val mono">${acc} / ${drf} <span style="font-size:12px;color:var(--muted);">(${accRate})</span></div>
        <div class="card-sub">Canceled ${can} · Run mean wait ${run.summary.mean_p0_queue_wait_ms.toFixed(2)} ms</div>
      </div>
    `;
  }

  let hitBoxes = [];

  function drawLanes() {
    const g = curGroup(), run = curRun(), cmp = cmpRun();
    const rowH = 34, topPad = 24, botPad = 10;
    const cssH = topPad + g.users.length * rowH + botPad;
    const { ctx, w, h, plotW } = setupCanvas(els.cvLanes, cssH);
    const [v0, v1] = viewWindow();
    hitBoxes = [];

    ctx.clearRect(0, 0, w, h);

    // Grid & time ticks
    const ticks = niceTicks(v0, v1, 8);
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ticks.forEach(tk => {
      const x = xFromTime(tk, plotW, v0, v1);
      ctx.strokeStyle = "#1e293b";
      ctx.beginPath(); ctx.moveTo(x, topPad - 4); ctx.lineTo(x, h - botPad); ctx.stroke();
      ctx.fillStyle = "#93a4bf";
      ctx.fillText((tk / 1000).toFixed(1) + "s", x, 14);
    });

    const cmpMap = new Map();
    if (cmp) cmp.turns.forEach(t => cmpMap.set(t.key, t));

    g.users.forEach((uid, idx) => {
      const y0 = topPad + idx * rowH;
      const cy = y0 + rowH / 2;
      ctx.fillStyle = idx % 2 === 0 ? "rgba(255,255,255,0.015)" : "transparent";
      ctx.fillRect(0, y0, w, rowH);
      ctx.strokeStyle = "#1e293b";
      ctx.beginPath(); ctx.moveTo(0, y0 + rowH); ctx.lineTo(w, y0 + rowH); ctx.stroke();

      // Gutter label + status pill
      const [stText, stFg, stBg] = userStatusAt(uid, run.turns, state.tMs, run.has_timeline);
      ctx.fillStyle = stBg;
      ctx.fillRect(6, cy - 9, 64, 18);
      ctx.strokeStyle = stFg;
      ctx.strokeRect(6, cy - 9, 64, 18);
      ctx.fillStyle = stFg;
      ctx.font = "bold 9.5px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(stText, 38, cy + 3);

      ctx.fillStyle = "#cbd5e1";
      ctx.font = "11.5px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(shortUser(uid), 76, cy + 4, GUTTER - 82);

      // Turns for this user
      ctx.save();
      ctx.beginPath();
      ctx.rect(GUTTER, y0, plotW + 6, rowH);
      ctx.clip();
      for (const tr of run.turns) {
        if (tr.conversation_id !== uid) continue;
        if (tr.completion_ms < v0 || tr.utterance_start_ms > v1) continue;

        const sx = xFromTime(tr.utterance_start_ms, plotW, v0, v1);
        const ex = xFromTime(tr.utterance_end_ms, plotW, v0, v1);
        const qx = xFromTime(tr.q_end_ms, plotW, v0, v1);
        const tx = xFromTime(tr.ttft_end_ms, plotW, v0, v1);
        const fx = xFromTime(tr.ttfs_end_ms, plotW, v0, v1);
        const cx = xFromTime(tr.completion_ms, plotW, v0, v1);
        const barH = 18, by = cy - barH / 2;

        // Speak bar
        if (ex > GUTTER) {
          const spX = Math.max(GUTTER, sx);
          const spW = Math.max(2, ex - spX);
          ctx.fillStyle = tr.is_warmup ? "rgba(59,130,246,0.34)" : "rgba(59,130,246,0.52)";
          ctx.fillRect(spX, by, spW, barH);
          ctx.strokeStyle = "#60a5fa";
          ctx.lineWidth = 1;
          ctx.strokeRect(spX, by, spW, barH);
        }

        // P1 draft bursts overlay
        if (run.has_timeline && tr.draft_bursts) {
          ctx.fillStyle = "#a855f7";
          for (const [bs, be] of tr.draft_bursts) {
            const bx0 = Math.max(GUTTER, xFromTime(bs, plotW, v0, v1));
            const bx1 = xFromTime(be, plotW, v0, v1);
            if (bx1 >= GUTTER) ctx.fillRect(bx0, by + barH - 5, Math.max(2, bx1 - bx0), 5);
          }
        }

        // Queue wait
        if (tr.q_end_ms > tr.utterance_end_ms + 0.1 && qx > GUTTER) {
          ctx.fillStyle = "#f59e0b";
          ctx.fillRect(Math.max(GUTTER, ex), by, Math.max(2, qx - Math.max(GUTTER, ex)), barH);
        }

        // GPU P0 block (guarantee >= 4px visibility when zoomed out)
        const gpuStart = Math.max(GUTTER, qx);
        const gpuW = Math.max(4, cx - gpuStart);
        if (cx > tx) {
          ctx.fillStyle = "#06b6d4";
          ctx.fillRect(gpuStart, by, Math.max(1.5, tx - gpuStart), barH);
        }
        const sloColor = tr.ttfs_ms <= g.slo_target_ms ? "#10b981" : "#ef4444";
        ctx.fillStyle = sloColor;
        const ttfsW = Math.max(2.5, fx - tx);
        ctx.fillRect(Math.max(gpuStart, tx), by, ttfsW, barH);
        if (cx > fx) {
          ctx.fillStyle = "#059669";
          ctx.fillRect(fx, by, Math.max(1, cx - fx), barH);
        }

        // Selection ring
        const totalRight = Math.min(GUTTER + plotW + 4, Math.max(gpuStart + gpuW, cx));
        if (tr.key === state.selectedKey) {
          ctx.strokeStyle = "#facc15";
          ctx.lineWidth = 2;
          ctx.strokeRect(Math.max(GUTTER, sx) - 1, by - 1, totalRight - Math.max(GUTTER, sx) + 2, barH + 2);
          ctx.lineWidth = 1;
        }

        // Compare policy completion tick
        const ctr = cmpMap.get(tr.key);
        if (ctr) {
          const cmpx = xFromTime(ctr.completion_ms, plotW, v0, v1);
          if (cmpx >= GUTTER && cmpx <= w - RPAD) {
            ctx.strokeStyle = "#c084fc";
            ctx.beginPath(); ctx.moveTo(cmpx, by - 3); ctx.lineTo(cmpx, by + barH + 3); ctx.stroke();
          }
        }

        // Label inside speak bar
        const availW = ex - Math.max(GUTTER, sx);
        if (availW > 38) {
          ctx.fillStyle = "#eff6ff";
          ctx.font = "10.5px ui-monospace, monospace";
          ctx.textAlign = "left";
          const tag = `t${tr.turn_id}${tr.synthetic_revision ? " REV" : ""}${tr.is_warmup ? " W" : ""}`;
          ctx.fillText(tag, Math.max(GUTTER, sx) + 4, cy + 3.5, availW - 6);
        }

        hitBoxes.push({
          x0: Math.max(GUTTER, sx),
          x1: Math.max(totalRight, Math.max(GUTTER, sx) + 6),
          y0: by,
          y1: by + barH,
          turn: tr,
        });
      }
      ctx.restore();
    });

    // Playhead
    if (state.tMs >= v0 && state.tMs <= v1) {
      const px = xFromTime(state.tMs, plotW, v0, v1);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, topPad - 6); ctx.lineTo(px, h - botPad); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  function drawGpuChart() {
    const g = curGroup(), run = curRun(), cmp = cmpRun();
    const topPad = 14, botPad = 22, cssH = 165;
    const { ctx, w, h, plotW } = setupCanvas(els.cvGpu, cssH);
    const [v0, v1] = viewWindow();
    const chartH = h - topPad - botPad;

    ctx.clearRect(0, 0, w, h);

    // Y gridlines (0, 50, 100%)
    ctx.font = "10.5px ui-monospace, monospace";
    [0, 50, 100].forEach(val => {
      const y = topPad + chartH * (1 - val / 100);
      ctx.strokeStyle = "#1e293b";
      ctx.beginPath(); ctx.moveTo(GUTTER, y); ctx.lineTo(GUTTER + plotW, y); ctx.stroke();
      ctx.fillStyle = "#93a4bf";
      ctx.textAlign = "right";
      ctx.fillText(val + "%", GUTTER - 8, y + 3.5);
    });

    const maxCnt = Math.max(2, g.users.length, run.aggregates.peak_batch, run.aggregates.peak_queue);
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "left";
    ctx.fillText("0", GUTTER + plotW + 4, topPad + chartH + 3);
    ctx.fillText(String(maxCnt), GUTTER + plotW + 4, topPad + 7);

    // Stacked GPU bars
    for (const b of run.bins) {
      if (b.t1 < v0 || b.t0 > v1) continue;
      const x0 = Math.max(GUTTER, xFromTime(b.t0, plotW, v0, v1));
      const x1 = Math.min(GUTTER + plotW, xFromTime(b.t1, plotW, v0, v1));
      const bw = Math.max(1, x1 - x0);
      const h0 = (b.p0_util_pct / 100) * chartH;
      const h1 = (b.p1_util_pct / 100) * chartH;
      ctx.fillStyle = "rgba(16, 185, 129, 0.58)";
      ctx.fillRect(x0, topPad + chartH - h0, bw, h0);
      if (h1 > 0.2) {
        ctx.fillStyle = "rgba(168, 85, 247, 0.62)";
        ctx.fillRect(x0, topPad + chartH - h0 - h1, bw, h1);
      }
    }

    // Compare GPU busy dashed outline
    if (cmp) {
      ctx.strokeStyle = "rgba(192, 132, 252, 0.75)";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (const b of cmp.bins) {
        if (b.t1 < v0 || b.t0 > v1) continue;
        const mx = xFromTime((b.t0 + b.t1) / 2, plotW, v0, v1);
        const my = topPad + chartH * (1 - b.total_gpu_busy_pct / 100);
        if (!started) { ctx.moveTo(mx, my); started = true; } else ctx.lineTo(mx, my);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Concurrency lines on right axis
    function drawCountLine(getter, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (const b of run.bins) {
        if (b.t1 < v0 || b.t0 > v1) continue;
        const mx = xFromTime((b.t0 + b.t1) / 2, plotW, v0, v1);
        const my = topPad + chartH * (1 - Math.min(maxCnt, getter(b)) / maxCnt);
        if (!started) { ctx.moveTo(mx, my); started = true; } else ctx.lineTo(mx, my);
      }
      ctx.stroke();
      ctx.lineWidth = 1;
    }
    drawCountLine(b => b.active_speakers, "#3b82f6");
    drawCountLine(b => b.max_batch_p0 + b.max_batch_p1, "#22d3ee");
    drawCountLine(b => b.max_queue_p0 + b.max_queue_p1, "#f59e0b");

    // Left panel live summary
    const curB = findBinAt(run, state.tMs);
    ctx.textAlign = "left";
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px ui-monospace, monospace";
    if (curB) {
      ctx.fillText(`GPU: ${curB.total_gpu_busy_pct.toFixed(1)}%`, 8, 28);
      ctx.fillStyle = "#34d399";
      ctx.fillText(`P0: ${curB.p0_util_pct.toFixed(1)}%`, 8, 46);
      ctx.fillStyle = "#c084fc";
      ctx.fillText(`P1: ${curB.p1_util_pct.toFixed(1)}%`, 8, 64);
      ctx.fillStyle = "#60a5fa";
      ctx.fillText(`Speak: ${curB.active_speakers}`, 8, 86);
      ctx.fillStyle = "#22d3ee";
      ctx.fillText(`Batch: ${curB.max_batch_p0 + curB.max_batch_p1}/${g.max_batch_size}`, 8, 104);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(`Queue: ${curB.max_queue_p0 + curB.max_queue_p1}`, 8, 122);
    }

    // Playhead
    if (state.tMs >= v0 && state.tMs <= v1) {
      const px = xFromTime(state.tMs, plotW, v0, v1);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, topPad); ctx.lineTo(px, topPad + chartH); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  let latPoints = [];

  function drawLatencyChart() {
    const g = curGroup(), run = curRun(), cmp = cmpRun();
    const topPad = 14, botPad = 24, cssH = 165;
    const { ctx, w, h, plotW } = setupCanvas(els.cvLat, cssH);
    const [v0, v1] = viewWindow();
    const chartH = h - topPad - botPad;
    latPoints = [];

    ctx.clearRect(0, 0, w, h);

    let maxMs = Math.max(g.slo_target_ms * 1.25, 50);
    for (const tr of run.turns) maxMs = Math.max(maxMs, tr.ttfs_ms * 1.12);
    if (cmp) for (const tr of cmp.turns) maxMs = Math.max(maxMs, tr.ttfs_ms * 1.12);

    ctx.font = "10.5px ui-monospace, monospace";
    [0, 0.5, 1].forEach(frac => {
      const val = maxMs * frac;
      const y = topPad + chartH * (1 - frac);
      ctx.strokeStyle = "#1e293b";
      ctx.beginPath(); ctx.moveTo(GUTTER, y); ctx.lineTo(GUTTER + plotW, y); ctx.stroke();
      ctx.fillStyle = "#93a4bf";
      ctx.textAlign = "right";
      ctx.fillText(val.toFixed(0) + "ms", GUTTER - 8, y + 3.5);
    });

    // SLO line
    const ySlo = topPad + chartH * (1 - g.slo_target_ms / maxMs);
    ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(GUTTER, ySlo); ctx.lineTo(GUTTER + plotW, ySlo); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f87171";
    ctx.textAlign = "left";
    ctx.fillText(`SLO ${g.slo_target_ms}ms`, 8, Math.max(22, Math.min(topPad + chartH - 4, ySlo + 3)));

    const sortedTurns = [...run.turns].sort((a, b) => a.utterance_end_ms - b.utterance_end_ms);

    // Compare TTFS dashed line
    if (cmp) {
      const cmpSorted = [...cmp.turns].sort((a, b) => a.utterance_end_ms - b.utterance_end_ms);
      ctx.strokeStyle = "#c084fc";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (const tr of cmpSorted) {
        if (tr.utterance_end_ms < v0 || tr.utterance_end_ms > v1) continue;
        const x = xFromTime(tr.utterance_end_ms, plotW, v0, v1);
        const y = topPad + chartH * (1 - tr.ttfs_ms / maxMs);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Queue wait bars
    ctx.fillStyle = "#f59e0b";
    for (const tr of sortedTurns) {
      if (tr.utterance_end_ms < v0 || tr.utterance_end_ms > v1 || tr.p0_queue_wait_ms <= 0.1) continue;
      const x = xFromTime(tr.utterance_end_ms, plotW, v0, v1);
      const y = topPad + chartH * (1 - tr.p0_queue_wait_ms / maxMs);
      ctx.fillRect(x - 1.5, y, 3, topPad + chartH - y);
    }

    // Main TTFS line
    ctx.strokeStyle = "rgba(16, 185, 129, 0.65)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (const tr of sortedTurns) {
      if (tr.utterance_end_ms < v0 || tr.utterance_end_ms > v1) continue;
      const x = xFromTime(tr.utterance_end_ms, plotW, v0, v1);
      const y = topPad + chartH * (1 - tr.ttfs_ms / maxMs);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineWidth = 1;

    // Points
    for (const tr of sortedTurns) {
      if (tr.utterance_end_ms < v0 || tr.utterance_end_ms > v1) continue;
      const x = xFromTime(tr.utterance_end_ms, plotW, v0, v1);
      const yTtfs = topPad + chartH * (1 - tr.ttfs_ms / maxMs);
      const yTtft = topPad + chartH * (1 - tr.ttft_ms / maxMs);

      ctx.fillStyle = "#06b6d4";
      ctx.beginPath(); ctx.arc(x, yTtft, 2.8, 0, Math.PI * 2); ctx.fill();

      const col = tr.ttfs_ms <= g.slo_target_ms ? "#10b981" : "#ef4444";
      ctx.beginPath(); ctx.arc(x, yTtfs, tr.key === state.selectedKey ? 5.5 : 4.2, 0, Math.PI * 2);
      if (tr.is_warmup) {
        ctx.fillStyle = "#0b101b"; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke(); ctx.lineWidth = 1;
      } else {
        ctx.fillStyle = col; ctx.fill();
      }
      if (tr.key === state.selectedKey) {
        ctx.strokeStyle = "#facc15"; ctx.lineWidth = 1.8; ctx.stroke(); ctx.lineWidth = 1;
      }
      latPoints.push({ x, y: yTtfs, turn: tr });
    }

    // Playhead
    if (state.tMs >= v0 && state.tMs <= v1) {
      const px = xFromTime(state.tMs, plotW, v0, v1);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, topPad); ctx.lineTo(px, topPad + chartH); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  function renderInspector() {
    const run = curRun(), cmp = cmpRun();
    const tr = run.turns.find(t => t.key === state.selectedKey) || run.turns[0];
    if (!tr) { els.inspector.textContent = "No turns available."; return; }
    state.selectedKey = tr.key;
    els.inspSub.textContent = `${shortUser(tr.conversation_id)} · t${tr.turn_id}`;

    const ctr = cmp ? cmp.turns.find(t => t.key === tr.key) : null;
    let cmpHtml = "";
    if (ctr) {
      const dt = tr.ttfs_ms - ctr.ttfs_ms;
      const cls = dt <= 0 ? "pos" : "neg";
      const matchStr = tr.response_text === ctr.response_text
        ? '<span class="pos">Response matches comparison verbatim</span>'
        : '<span style="color:#fbbf24;">Response differs (e.g. batching numeric precision)</span>';
      cmpHtml = `
        <div class="kv"><div class="k">vs ${esc(cmp.policy_name)} TTFS</div><div class="v mono ${cls}">${dt > 0 ? "+" : ""}${dt.toFixed(1)} ms</div></div>
        <div style="margin:6px 0;font-size:11.5px;">${matchStr}</div>
        ${tr.response_text !== ctr.response_text ? `<div class="resp-box" style="margin-top:4px;border-color:#6d28d9;"><b>${esc(cmp.policy_name)}:</b>\n${esc(ctr.response_text)}</div>` : ""}
      `;
    }

    els.inspector.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
        <span class="pill ${tr.ttfs_ms <= curGroup().slo_target_ms ? "ok" : "miss"}">${tr.ttfs_ms <= curGroup().slo_target_ms ? "SLO OK" : "SLO MISS"}</span>
        <span class="badge">${esc(tr.family)} · ${esc(tr.pacing)}</span>
        ${tr.synthetic_revision ? '<span class="badge">synthetic_revision</span>' : ""}
        ${tr.is_warmup ? '<span class="pill warm">warmup</span>' : ""}
      </div>
      <div class="kv-grid">
        <div class="kv"><div class="k">Speaking Window</div><div class="v mono">${(tr.utterance_start_ms/1000).toFixed(2)}s - ${(tr.utterance_end_ms/1000).toFixed(2)}s</div></div>
        <div class="kv"><div class="k">P0 Queue Wait</div><div class="v mono">${tr.p0_queue_wait_ms.toFixed(2)} ms</div></div>
        <div class="kv"><div class="k">TTFT / TTFS</div><div class="v mono">${tr.ttft_ms.toFixed(1)} / ${tr.ttfs_ms.toFixed(1)} ms</div></div>
        <div class="kv"><div class="k">Total Response</div><div class="v mono">${tr.total_response_ms.toFixed(1)} ms</div></div>
        <div class="kv"><div class="k">Draft Acc / Tot</div><div class="v mono">${tr.accepted_tokens} / ${tr.drafted_tokens} (${(tr.acceptance_rate*100).toFixed(1)}%)</div></div>
        <div class="kv"><div class="k">Canceled Draft</div><div class="v mono">${tr.canceled_tokens} tok</div></div>
      </div>
      <div style="color:var(--muted);font-size:11px;margin-bottom:4px;"><b>${esc(run.policy_name)} Response:</b></div>
      <div class="resp-box">${esc(tr.response_text)}</div>
      ${cmpHtml}
    `;
  }

  function renderTable() {
    const g = curGroup(), run = curRun(), cmp = cmpRun();
    els.tblSub.textContent = `${run.turns.length} turns (${run.summary.warmup_turns} warmup)`;
    els.thDelta.textContent = cmp ? `vs ${cmp.policy_name}` : "Drafts";
    const cmpMap = new Map();
    if (cmp) cmp.turns.forEach(t => cmpMap.set(t.key, t));

    els.tblBody.innerHTML = "";
    for (const tr of run.turns) {
      const row = document.createElement("tr");
      if (tr.key === state.selectedKey) row.className = "selected";
      const ctr = cmpMap.get(tr.key);
      let deltaCell = `${tr.canceled_tokens} can`;
      if (ctr) {
        const dt = tr.ttfs_ms - ctr.ttfs_ms;
        deltaCell = `<span class="${dt <= 0 ? "pos" : "neg"}">${dt > 0 ? "+" : ""}${dt.toFixed(1)}</span>`;
      }
      const statusPill = tr.is_warmup
        ? '<span class="pill warm">warmup</span>'
        : (tr.ttfs_ms <= g.slo_target_ms ? '<span class="pill ok">OK</span>' : '<span class="pill miss">&gt;SLO</span>');

      row.innerHTML = `
        <td class="mono">${esc(shortUser(tr.conversation_id))}</td>
        <td class="mono">${tr.turn_id}</td>
        <td>${esc(tr.family)}</td>
        <td class="mono">${(tr.utterance_end_ms / 1000).toFixed(2)}</td>
        <td class="mono">${tr.p0_queue_wait_ms.toFixed(1)}</td>
        <td class="mono">${tr.ttft_ms.toFixed(1)}</td>
        <td class="mono"><b>${tr.ttfs_ms.toFixed(1)}</b></td>
        <td class="mono">${tr.total_response_ms.toFixed(1)}</td>
        <td class="mono">${tr.accepted_tokens}/${tr.drafted_tokens}</td>
        <td class="mono">${deltaCell}</td>
        <td>${statusPill}</td>
      `;
      row.addEventListener("click", () => {
        state.selectedKey = tr.key;
        state.tMs = tr.utterance_end_ms;
        state.playing = false;
        updateAll();
      });
      els.tblBody.appendChild(row);
    }
  }

  function updateClockAndBadge() {
    const g = curGroup(), run = curRun();
    els.btnPlay.textContent = state.playing ? "Pause" : "Play";
    els.lblClock.textContent = `${(state.tMs / 1000).toFixed(2)} / ${(g.t_max_ms / 1000).toFixed(2)} s`;
    els.rngTime.value = String(Math.round(state.tMs));
    if (run.has_timeline) {
      els.badge.className = "badge full";
      els.badge.textContent = "FULL TIMELINE MODE (Timed calls; P0/P1 split estimated by sequence count)";
    } else {
      els.badge.className = "badge turn";
      els.badge.textContent = "TURN-ONLY MODE (Foreground response envelopes; GPU occupancy estimated)";
    }
  }

  function redrawDynamic() {
    updateClockAndBadge();
    renderCards();
    drawLanes();
    drawGpuChart();
    drawLatencyChart();
  }

  function updateAll() {
    redrawDynamic();
    renderInspector();
    renderTable();
  }

  // Tooltip + canvas scrubbing / turn click
  function showTip(evt, htmlStr) {
    els.tooltip.innerHTML = htmlStr;
    els.tooltip.style.display = "block";
    els.tooltip.style.left = Math.min(window.innerWidth - 380, evt.clientX + 14) + "px";
    els.tooltip.style.top = Math.min(window.innerHeight - 160, evt.clientY + 14) + "px";
  }
  function hideTip() { els.tooltip.style.display = "none"; }

  function bindCanvasScrub(cv, kind) {
    let dragging = false, panStartX = 0, panView0 = 0;
    cv.addEventListener("mousedown", (e) => {
      const rect = cv.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      if (kind === "lanes") {
        const hit = hitBoxes.find(b => x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1);
        if (hit) {
          state.selectedKey = hit.turn.key;
          state.tMs = hit.turn.utterance_end_ms;
          state.playing = false;
          updateAll();
          return;
        }
      } else if (kind === "lat") {
        const hit = latPoints.find(p => Math.hypot(p.x - x, p.y - y) <= 8);
        if (hit) {
          state.selectedKey = hit.turn.key;
          state.tMs = hit.turn.utterance_end_ms;
          state.playing = false;
          updateAll();
          return;
        }
      }
      dragging = true;
      panStartX = x;
      panView0 = state.viewStartMs;
      if (!e.shiftKey) {
        const plotW = Math.max(100, cv.clientWidth - GUTTER - RPAD);
        const [v0, v1] = viewWindow();
        if (x >= GUTTER) {
          state.tMs = Math.max(0, Math.min(curGroup().t_max_ms, timeFromX(x, plotW, v0, v1)));
          redrawDynamic();
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const rect = cv.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const plotW = Math.max(100, cv.clientWidth - GUTTER - RPAD);
      const [v0, v1] = viewWindow();
      if (e.shiftKey && state.zoom > 1) {
        const dx = x - panStartX;
        const dt = (dx / plotW) * (v1 - v0);
        state.viewStartMs = Math.max(0, Math.min(curGroup().t_max_ms - (v1 - v0), panView0 - dt));
      } else if (x >= GUTTER) {
        state.tMs = Math.max(0, Math.min(curGroup().t_max_ms, timeFromX(x, plotW, v0, v1)));
      }
      redrawDynamic();
    });
    window.addEventListener("mouseup", () => { dragging = false; });

    cv.addEventListener("mousemove", (e) => {
      if (dragging) return;
      const rect = cv.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      if (x < GUTTER) { hideTip(); return; }
      const plotW = Math.max(100, cv.clientWidth - GUTTER - RPAD);
      const [v0, v1] = viewWindow();
      const tHover = timeFromX(x, plotW, v0, v1);

      if (kind === "lanes") {
        const hit = hitBoxes.find(b => x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1);
        if (hit) {
          const t = hit.turn;
          showTip(e, `<b>${esc(t.conversation_id)} (t${t.turn_id})</b><br>` +
            `${esc(t.family)} · ${esc(t.pacing)} ${t.is_warmup ? "· WARMUP" : ""}<br>` +
            `Speak: ${(t.utterance_start_ms/1000).toFixed(2)}s - ${(t.utterance_end_ms/1000).toFixed(2)}s<br>` +
            `Wait: <b>${t.p0_queue_wait_ms.toFixed(1)} ms</b> · TTFT: ${t.ttft_ms.toFixed(1)} ms<br>` +
            `TTFS: <b>${t.ttfs_ms.toFixed(1)} ms</b> · Total: ${t.total_response_ms.toFixed(1)} ms<br>` +
            `Drafts: ${t.accepted_tokens}/${t.drafted_tokens} acc · ${t.canceled_tokens} canceled`);
        } else {
          showTip(e, `t = <b>${(tHover/1000).toFixed(2)} s</b> (${tHover.toFixed(0)} ms)`);
        }
      } else if (kind === "gpu") {
        const b = findBinAt(curRun(), tHover);
        if (b) {
          showTip(e, `<b>Window ${(b.t0/1000).toFixed(2)}s - ${(b.t1/1000).toFixed(2)}s</b><br>` +
            `Scheduled Busy Proxy: <b>${b.total_gpu_busy_pct.toFixed(1)}%</b> (P0 ${b.p0_util_pct.toFixed(1)}%, P1 ${b.p1_util_pct.toFixed(1)}%)<br>` +
            `Peak Speakers: ${b.active_speakers} · Batch: ${b.max_batch_p0}+${b.max_batch_p1} · Queue: ${b.max_queue_p0}`);
        }
      } else if (kind === "lat") {
        const hit = latPoints.find(p => Math.hypot(p.x - x, p.y - y) <= 8);
        if (hit) {
          const t = hit.turn;
          showTip(e, `<b>${esc(shortUser(t.conversation_id))} t${t.turn_id}</b><br>` +
            `TTFS: <b>${t.ttfs_ms.toFixed(1)} ms</b> · TTFT: ${t.ttft_ms.toFixed(1)} ms · Wait: ${t.p0_queue_wait_ms.toFixed(1)} ms`);
        } else {
          hideTip();
        }
      }
    });
    cv.addEventListener("mouseleave", hideTip);
  }

  bindCanvasScrub(els.cvLanes, "lanes");
  bindCanvasScrub(els.cvGpu, "gpu");
  bindCanvasScrub(els.cvLat, "lat");

  els.selGroup.addEventListener("change", () => { state.groupIdx = Number(els.selGroup.value); state.playing = false; populatePolicies(); });
  els.selPolicy.addEventListener("change", () => { state.policy = els.selPolicy.value; updateAll(); });
  els.selCompare.addEventListener("change", () => { state.compare = els.selCompare.value; updateAll(); });
  els.selSpeed.addEventListener("change", () => { state.speed = Number(els.selSpeed.value); });
  els.selZoom.addEventListener("change", () => { state.zoom = Number(els.selZoom.value); redrawDynamic(); });
  els.rngTime.addEventListener("input", () => { state.tMs = Number(els.rngTime.value); state.playing = false; redrawDynamic(); });

  els.btnPlay.addEventListener("click", () => {
    if (state.tMs >= curGroup().t_max_ms - 1) state.tMs = 0;
    state.playing = !state.playing;
    state.lastTs = null;
    redrawDynamic();
  });
  els.btnReset.addEventListener("click", () => { state.playing = false; state.tMs = 0; state.viewStartMs = 0; redrawDynamic(); });

  function stepTurn(delta) {
    const turns = [...curRun().turns].sort((a, b) => a.utterance_end_ms - b.utterance_end_ms);
    if (!turns.length) return;
    let idx = turns.findIndex(t => t.key === state.selectedKey);
    idx = idx < 0 ? 0 : Math.max(0, Math.min(turns.length - 1, idx + delta));
    state.selectedKey = turns[idx].key;
    state.tMs = turns[idx].utterance_end_ms;
    state.playing = false;
    updateAll();
  }
  els.btnPrev.addEventListener("click", () => stepTurn(-1));
  els.btnNext.addEventListener("click", () => stepTurn(1));

  window.addEventListener("keydown", (e) => {
    if (e.target && ["SELECT", "INPUT"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); els.btnPlay.click(); }
    else if (e.key === "[") stepTurn(-1);
    else if (e.key === "]") stepTurn(1);
  });
  window.addEventListener("resize", redrawDynamic);

  function tick(ts) {
    if (state.playing) {
      if (state.lastTs !== null) {
        const dt = Math.min(100, ts - state.lastTs) * state.speed;
        state.tMs += dt;
        const tMax = curGroup().t_max_ms;
        if (state.tMs >= tMax) { state.tMs = tMax; state.playing = false; }
      }
      state.lastTs = ts;
      redrawDynamic();
    } else {
      state.lastTs = null;
    }
    requestAnimationFrame(tick);
  }

  populateGroups();
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""


def render_html_dashboard(payload: Dict[str, Any]) -> str:
    """Render self-contained interactive HTML5/Canvas simulation dashboard."""
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(payload.get("source_name") or "Experiment Replay"))
    return (
        _HTML_TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD_JSON__", safe_json)
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive HTML simulation replay (and optional ASCII view) from experiment outputs."
    )
    parser.add_argument(
        "input_path",
        help="Path to turns_*.jsonl or summaries_*.json",
    )
    parser.add_argument("--summaries", default=None, help="Optional summaries_*.json override path")
    parser.add_argument("--timeline", default=None, help="Optional timeline_*.jsonl override path")
    parser.add_argument("--output", default=None, help="Output HTML path (default: sibling replay_<suffix>.html)")
    parser.add_argument("--bin_ms", type=float, default=None, help="Time bin width in ms (default: auto 100..1000ms)")
    parser.add_argument("--max_batch_size", type=int, default=8, help="Fallback max batch size if no timeline exists")
    parser.add_argument("--slo_target_ms", type=float, default=190.0, help="Fallback SLO target ms if no summaries exist")
    parser.add_argument("--ascii", action="store_true", help="Also print ASCII timeline and summary table to stdout")
    args = parser.parse_args(argv)

    turns_path, summaries_path, timeline_path, out_path = resolve_input_paths(
        Path(args.input_path),
        summaries_arg=args.summaries,
        timeline_arg=args.timeline,
        output_arg=args.output,
    )

    turns_rows = _load_jsonl(turns_path)
    summary_rows = _load_summaries(summaries_path)
    timeline_rows = _load_jsonl(timeline_path) if timeline_path else []

    payload = build_visualization_payload(
        turns_rows=turns_rows,
        summary_rows=summary_rows,
        timeline_rows=timeline_rows,
        bin_ms=args.bin_ms,
        max_batch_size=args.max_batch_size,
        slo_target_ms=args.slo_target_ms,
        source_name=turns_path.name,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html_dashboard(payload), encoding="utf-8")

    if args.ascii:
        print(format_ascii_report(payload))
        print("-" * 88)
    print(f"Generated interactive simulation replay: {out_path}")
    if timeline_path:
        print(f"Loaded step-level GPU timeline:        {timeline_path}")
    else:
        print("Timeline JSONL not found; rendered turn-only mode (foreground envelope estimate).")


if __name__ == "__main__":
    main()
