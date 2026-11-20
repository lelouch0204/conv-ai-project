"""Deterministic Poisson workload schedule generator for paired policy replay."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.data.schema import TurnTrace

DEFAULT_LOAD_RATES_RPS: Dict[str, float] = {
    "light": 0.25,
    "intermediate": 1.6,
    "heavy": 4.2,
}


@dataclass
class ArrivalEvent:
    """Scheduled start of one spoken turn in a replay workload."""

    utterance_start_ms: float
    stream_conversation_id: str
    trace: TurnTrace


def generate_workload_schedule(
    traces: Sequence[TurnTrace],
    arrival_rate_rps: float,
    seed: int = 42,
    target_turns: Optional[int] = None,
    inter_turn_gap_ms: float = 1100.0,
) -> List[ArrivalEvent]:
    """Generate a deterministic multi-speaker Poisson arrival schedule.

    Grouping by ``conversation_id`` preserves multi-turn history and cross-turn
    EMA state per stream while allowing concurrent conversations to overlap
    at the requested ``arrival_rate_rps``.
    """
    if arrival_rate_rps <= 0:
        raise ValueError(f"arrival_rate_rps must be > 0, got {arrival_rate_rps}")
    if not traces:
        return []

    by_conv: Dict[str, List[TurnTrace]] = defaultdict(list)
    for trace in traces:
        by_conv[trace.conversation_id].append(trace)

    conv_groups: List[List[TurnTrace]] = []
    for conv_id in sorted(by_conv.keys()):
        turns = sorted(by_conv[conv_id], key=lambda t: t.turn_id)
        conv_groups.append(turns)

    avg_turns_per_conv = sum(len(g) for g in conv_groups) / len(conv_groups)
    conv_rate_per_ms = (arrival_rate_rps / max(1.0, avg_turns_per_conv)) / 1000.0

    rng = random.Random(seed)
    goal_turns = target_turns if target_turns is not None else len(traces)

    # Build shuffled pool of conversation indices and sample streams until goal_turns met
    schedule: List[ArrivalEvent] = []
    current_conv_start_ms = 0.0
    stream_idx = 0
    pool_indices: List[int] = []

    while len(schedule) < goal_turns:
        if not pool_indices:
            pool_indices = list(range(len(conv_groups)))
            rng.shuffle(pool_indices)
        group = conv_groups[pool_indices.pop()]
        base_conv_id = group[0].conversation_id
        stream_conv_id = f"{base_conv_id}#s{stream_idx}"

        if stream_idx > 0:
            current_conv_start_ms += rng.expovariate(conv_rate_per_ms)

        turn_start_ms = current_conv_start_ms
        for turn in group:
            if len(schedule) >= goal_turns:
                break
            schedule.append(
                ArrivalEvent(
                    utterance_start_ms=round(turn_start_ms, 2),
                    stream_conversation_id=stream_conv_id,
                    trace=turn,
                )
            )
            turn_start_ms += turn.final.offset_ms + inter_turn_gap_ms

        stream_idx += 1

    schedule.sort(
        key=lambda ev: (
            ev.utterance_start_ms,
            ev.stream_conversation_id,
            ev.trace.turn_id,
        )
    )
    return schedule
