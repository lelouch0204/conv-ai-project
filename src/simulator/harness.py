"""Discrete-event multi-client replay harness and telemetry aggregator."""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from src.controllers.base import BaseController
from src.engine.base import BaseLLMEngine
from src.scheduler.gateway import PriorityGateway, TurnCompletionRecord
from src.simulator.workload import ArrivalEvent


@dataclass
class TurnMetrics:
    """Turn-level telemetry emitted by ReplayHarness."""

    policy_name: str
    load_regime: str
    seed: int
    conversation_id: str
    turn_id: int
    family: str
    pacing: str
    synthetic_revision: bool
    utterance_start_ms: float
    utterance_end_ms: float
    completion_ms: float
    ttft_ms: float
    ttfs_ms: float
    total_response_ms: float
    p0_queue_wait_ms: float
    drafted_tokens: int
    accepted_tokens: int
    canceled_tokens: int
    acceptance_rate: float
    timed_out: bool
    is_warmup: bool
    response_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Aggregated policy performance summary for one workload run."""

    policy_name: str
    load_regime: str
    arrival_rate_rps: float
    seed: int
    num_turns: int
    warmup_turns: int
    ttfs_p50_ms: float
    ttfs_p95_ms: float
    ttfs_mean_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    slo_target_ms: float
    slo_attainment_pct: float
    goodput_turns_per_sec: float
    goodput_tokens_per_sec: float
    total_drafted_tokens: int
    total_accepted_tokens: int
    total_canceled_tokens: int
    mean_acceptance_rate: float
    mean_p0_queue_wait_ms: float
    timeout_count: int
    total_sim_duration_ms: float
    by_family: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReplayHarness:
    """Event-driven simulator coupling a paired workload schedule with PriorityGateway."""

    def __init__(
        self,
        engine: BaseLLMEngine,
        slo_target_ms: float = 250.0,
        timeout_ms: float = 2500.0,
        max_batch_size: int = 8,
        max_new_tokens: int = 48,
        warmup_turns: int = 0,
        record_timeline: bool = False,
    ) -> None:
        self.engine = engine
        self.slo_target_ms = float(slo_target_ms)
        self.timeout_ms = float(timeout_ms)
        self.max_batch_size = int(max_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.warmup_turns = max(0, int(warmup_turns))
        self.record_timeline = bool(record_timeline)
        self.last_timeline: List[Dict[str, Any]] = []

    def run(
        self,
        schedule: Sequence[ArrivalEvent],
        controller: BaseController,
        load_regime: str = "custom",
        arrival_rate_rps: float = 1.0,
        seed: int = 42,
    ) -> Tuple[List[TurnMetrics], RunSummary]:
        """Replay ``schedule`` under ``controller`` and return turn metrics + summary."""
        gateway = PriorityGateway(
            engine=self.engine,
            controller=controller,
            max_batch_size=self.max_batch_size,
            max_new_tokens=self.max_new_tokens,
            record_trace=self.record_timeline,
        )

        # Build min-heap of (event_time_ms, seq_id, event_kind, payload)
        event_heap: List[Tuple[float, int, str, Dict[str, Any]]] = []
        seq_counter = 0
        for ev in schedule:
            trace = ev.trace
            for partial in trace.partials:
                heapq.heappush(
                    event_heap,
                    (
                        ev.utterance_start_ms + partial.offset_ms,
                        seq_counter,
                        "partial",
                        {
                            "conversation_id": ev.stream_conversation_id,
                            "turn_id": trace.turn_id,
                            "history": trace.history,
                            "text": partial.text,
                            "utterance_start_ms": ev.utterance_start_ms,
                        },
                    ),
                )
                seq_counter += 1

            heapq.heappush(
                event_heap,
                (
                    ev.utterance_start_ms + trace.final.offset_ms,
                    seq_counter,
                    "final",
                    {
                        "conversation_id": ev.stream_conversation_id,
                        "turn_id": trace.turn_id,
                        "history": trace.history,
                        "text": trace.final.text,
                        "utterance_start_ms": ev.utterance_start_ms,
                        "utterance_end_ms": ev.utterance_start_ms + trace.final.offset_ms,
                        "family": trace.family,
                        "pacing": trace.pacing,
                        "synthetic_revision": trace.synthetic_revision,
                    },
                ),
            )
            seq_counter += 1

        completions: List[TurnCompletionRecord] = []
        now_ms = 0.0

        while event_heap or gateway.has_work():
            while event_heap and event_heap[0][0] <= now_ms + 1e-9:
                ev_time, _, kind, payload = heapq.heappop(event_heap)
                if kind == "partial":
                    gateway.on_partial_arrive(
                        conversation_id=payload["conversation_id"],
                        turn_id=payload["turn_id"],
                        history=payload["history"],
                        partial_text=payload["text"],
                        utterance_start_ms=payload["utterance_start_ms"],
                        now_ms=ev_time,
                    )
                else:
                    gateway.on_final_arrive(
                        conversation_id=payload["conversation_id"],
                        turn_id=payload["turn_id"],
                        history=payload["history"],
                        final_text=payload["text"],
                        utterance_start_ms=payload["utterance_start_ms"],
                        utterance_end_ms=payload["utterance_end_ms"],
                        family=payload["family"],
                        pacing=payload["pacing"],
                        synthetic_revision=payload["synthetic_revision"],
                    )

            if gateway.has_work():
                step_ms, newly_completed = gateway.step(now_ms)
                now_ms += max(0.1, step_ms)
                completions.extend(newly_completed)
            elif event_heap:
                now_ms = event_heap[0][0]

        if self.record_timeline:
            self.last_timeline = [
                {
                    "policy_name": controller.name,
                    "load_regime": load_regime,
                    "seed": seed,
                    "t0": round(s.start_ms, 3),
                    "dt": round(s.step_ms, 3),
                    "k": s.kind,
                    "p0": s.p0_active,
                    "p1": s.p1_active,
                    "q0": s.p0_queue_len,
                    "q1": s.p1_queue_len,
                    "cap": s.max_batch_size,
                    "c0": s.p0_convs,
                    "c1": s.p1_convs,
                }
                for s in gateway.step_trace
            ]
        else:
            self.last_timeline = []

        turn_metrics: List[TurnMetrics] = []
        for idx, rec in enumerate(completions):
            turn_metrics.append(
                TurnMetrics(
                    policy_name=controller.name,
                    load_regime=load_regime,
                    seed=seed,
                    conversation_id=rec.conversation_id,
                    turn_id=rec.turn_id,
                    family=rec.family,
                    pacing=rec.pacing,
                    synthetic_revision=rec.synthetic_revision,
                    utterance_start_ms=rec.utterance_start_ms,
                    utterance_end_ms=rec.utterance_end_ms,
                    completion_ms=rec.completion_ms,
                    ttft_ms=rec.ttft_ms,
                    ttfs_ms=rec.ttfs_ms,
                    total_response_ms=rec.total_response_ms,
                    p0_queue_wait_ms=rec.p0_queue_wait_ms,
                    drafted_tokens=rec.drafted_tokens,
                    accepted_tokens=rec.accepted_tokens,
                    canceled_tokens=rec.canceled_tokens,
                    acceptance_rate=rec.acceptance_rate,
                    timed_out=rec.ttfs_ms > self.timeout_ms,
                    is_warmup=idx < self.warmup_turns,
                    response_text=rec.response_text,
                )
            )

        summary = self._summarize(
            completions=completions,
            metrics=turn_metrics,
            policy_name=controller.name,
            load_regime=load_regime,
            arrival_rate_rps=arrival_rate_rps,
            seed=seed,
        )
        return turn_metrics, summary

    def _summarize(
        self,
        completions: Sequence[TurnCompletionRecord],
        metrics: Sequence[TurnMetrics],
        policy_name: str,
        load_regime: str,
        arrival_rate_rps: float,
        seed: int,
    ) -> RunSummary:
        if not metrics:
            return RunSummary(
                policy_name=policy_name,
                load_regime=load_regime,
                arrival_rate_rps=arrival_rate_rps,
                seed=seed,
                num_turns=0,
                warmup_turns=0,
                ttfs_p50_ms=0.0,
                ttfs_p95_ms=0.0,
                ttfs_mean_ms=0.0,
                ttft_p50_ms=0.0,
                ttft_p95_ms=0.0,
                slo_target_ms=self.slo_target_ms,
                slo_attainment_pct=0.0,
                goodput_turns_per_sec=0.0,
                goodput_tokens_per_sec=0.0,
                total_drafted_tokens=0,
                total_accepted_tokens=0,
                total_canceled_tokens=0,
                mean_acceptance_rate=0.0,
                mean_p0_queue_wait_ms=0.0,
                timeout_count=0,
                total_sim_duration_ms=0.0,
                by_family={},
            )

        steady_pairs = [
            (c, m) for c, m in zip(completions, metrics) if not m.is_warmup
        ]
        if steady_pairs:
            eval_completions = [c for c, _ in steady_pairs]
            eval_metrics = [m for _, m in steady_pairs]
        else:
            eval_completions = list(completions)
            eval_metrics = list(metrics)

        warmup_count = sum(1 for m in metrics if m.is_warmup)
        ttfs_arr = np.array([m.ttfs_ms for m in eval_metrics], dtype=float)
        ttft_arr = np.array([m.ttft_ms for m in eval_metrics], dtype=float)
        wait_arr = np.array([m.p0_queue_wait_ms for m in eval_metrics], dtype=float)

        min_end = min(m.utterance_end_ms for m in eval_metrics)
        max_comp = max(m.completion_ms for m in eval_metrics)
        duration_ms = max(1.0, max_comp - min_end)
        duration_s = duration_ms / 1000.0

        slo_hits = sum(
            1 for m in eval_metrics if not m.timed_out and m.ttfs_ms <= self.slo_target_ms
        )
        total_drafted = sum(m.drafted_tokens for m in eval_metrics)
        total_accepted = sum(m.accepted_tokens for m in eval_metrics)
        total_canceled = sum(m.canceled_tokens for m in eval_metrics)
        total_valid_tokens = sum(len(c.response_tokens) for c in eval_completions)

        by_family_raw: Dict[str, List[TurnMetrics]] = defaultdict(list)
        for m in eval_metrics:
            by_family_raw[m.family].append(m)

        by_family: Dict[str, Dict[str, float]] = {}
        for fam, fam_list in sorted(by_family_raw.items()):
            fam_ttfs = np.array([x.ttfs_ms for x in fam_list], dtype=float)
            fam_drafted = sum(x.drafted_tokens for x in fam_list)
            fam_accepted = sum(x.accepted_tokens for x in fam_list)
            by_family[fam] = {
                "count": float(len(fam_list)),
                "ttfs_p50_ms": round(float(np.percentile(fam_ttfs, 50)), 2),
                "ttfs_p95_ms": round(float(np.percentile(fam_ttfs, 95)), 2),
                "acceptance_rate": round(
                    (fam_accepted / fam_drafted) if fam_drafted > 0 else 0.0, 4
                ),
            }

        return RunSummary(
            policy_name=policy_name,
            load_regime=load_regime,
            arrival_rate_rps=arrival_rate_rps,
            seed=seed,
            num_turns=len(eval_metrics),
            warmup_turns=warmup_count,
            ttfs_p50_ms=round(float(np.percentile(ttfs_arr, 50)), 2),
            ttfs_p95_ms=round(float(np.percentile(ttfs_arr, 95)), 2),
            ttfs_mean_ms=round(float(np.mean(ttfs_arr)), 2),
            ttft_p50_ms=round(float(np.percentile(ttft_arr, 50)), 2),
            ttft_p95_ms=round(float(np.percentile(ttft_arr, 95)), 2),
            slo_target_ms=self.slo_target_ms,
            slo_attainment_pct=round(100.0 * slo_hits / len(eval_metrics), 2),
            goodput_turns_per_sec=round(slo_hits / duration_s, 3),
            goodput_tokens_per_sec=round(total_valid_tokens / duration_s, 2),
            total_drafted_tokens=total_drafted,
            total_accepted_tokens=total_accepted,
            total_canceled_tokens=total_canceled,
            mean_acceptance_rate=round(
                (total_accepted / total_drafted) if total_drafted > 0 else 0.0, 4
            ),
            mean_p0_queue_wait_ms=round(float(np.mean(wait_arr)), 2),
            timeout_count=sum(1 for m in eval_metrics if m.timed_out),
            total_sim_duration_ms=round(duration_ms, 2),
            by_family=by_family,
        )


def _write_jsonl(rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_turn_metrics_jsonl(
    metrics: Sequence[TurnMetrics],
    path: Union[str, Path],
) -> None:
    """Persist turn-level telemetry records as JSONL."""
    _write_jsonl([row.to_dict() for row in metrics], path)


def save_timeline_jsonl(
    events: Sequence[Dict[str, Any]],
    path: Union[str, Path],
) -> None:
    """Persist GPU step-level timeline records as JSONL."""
    _write_jsonl(list(events), path)


def save_summaries_json(
    summaries: Sequence[RunSummary],
    path: Union[str, Path],
) -> None:
    """Persist aggregated run summaries as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in summaries], fh, indent=2)
