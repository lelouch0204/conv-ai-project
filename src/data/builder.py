"""Streaming trace builder and sample Full-Duplex-Bench-v2 style generator."""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.data.schema import (
    PartialChunk,
    TurnTrace,
    save_traces,
    validate_split_isolation,
)

_DEFAULT_SCENARIOS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw_scenarios.json"
)


@lru_cache(maxsize=4)
def _load_raw_scenarios(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_RAW_SCENARIOS: Dict[str, List[Dict[str, Any]]] = _load_raw_scenarios(
    _DEFAULT_SCENARIOS_PATH
)


def convert_raw_dialogue_to_traces(
    conversation_id: str,
    turns: Sequence[Dict[str, Any]],
    family: str = "daily",
    pacing: str = "fast",
    synthetic_revision: bool = False,
    rng: Optional[random.Random] = None,
) -> List[TurnTrace]:
    """Convert raw dialogue turns into timestamped partial-transcript TurnTraces.

    Each item in ``turns`` expects:
      - ``user`` (str): final utterance text (or pre-revision + pivot text)
      - ``assistant`` (str, optional): prior ground-truth assistant reply added to history
      - ``pre_revision_prefix`` (str, optional): spoken words before a self-correction
      - ``pause_after_word`` (int, optional): index where a mid-utterance pause occurs
    """
    gen = rng or random.Random(0)
    ms_per_word = 240.0 if pacing == "fast" else 420.0
    chunk_step_words = 3 if pacing == "fast" else 2
    history: List[Dict[str, str]] = []
    traces: List[TurnTrace] = []

    for turn_idx, item in enumerate(turns):
        final_text = str(item["user"]).strip()
        pre_rev = item.get("pre_revision_prefix")
        pause_after_word = item.get("pause_after_word")
        partials: List[PartialChunk] = []
        elapsed_ms = 0.0

        if pre_rev:
            pre_words = str(pre_rev).strip().split()
            for end in range(chunk_step_words, len(pre_words) + 1, chunk_step_words):
                elapsed_ms += chunk_step_words * ms_per_word + gen.uniform(-25.0, 25.0)
                elapsed_ms = max(elapsed_ms, 80.0)
                partials.append(
                    PartialChunk(offset_ms=round(elapsed_ms, 1), text=" ".join(pre_words[:end]))
                )
            if len(pre_words) % chunk_step_words != 0:
                elapsed_ms += (len(pre_words) % chunk_step_words) * ms_per_word
                partials.append(
                    PartialChunk(offset_ms=round(elapsed_ms, 1), text=" ".join(pre_words))
                )
            # Mid-utterance pause before self-correction arrives
            elapsed_ms += 650.0 if pacing == "fast" else 950.0
            final_words = final_text.split()
            # Reveal post-correction tail incrementally
            start_idx = min(len(pre_words) + 1, len(final_words))
            for end in range(start_idx, len(final_words) + 1, chunk_step_words):
                elapsed_ms += chunk_step_words * ms_per_word
                partials.append(
                    PartialChunk(offset_ms=round(elapsed_ms, 1), text=" ".join(final_words[:end]))
                )
            if not partials or partials[-1].text != " ".join(final_words):
                elapsed_ms += 180.0
                partials.append(
                    PartialChunk(offset_ms=round(elapsed_ms, 1), text=final_text)
                )
            final_offset = elapsed_ms + (140.0 if pacing == "fast" else 220.0)
            is_synthetic_rev = True
        else:
            words = final_text.split()
            idx = chunk_step_words
            while idx < len(words):
                elapsed_ms += chunk_step_words * ms_per_word + gen.uniform(-20.0, 20.0)
                elapsed_ms = max(elapsed_ms, 80.0)
                if pause_after_word is not None and idx >= int(pause_after_word):
                    elapsed_ms += 500.0 if pacing == "fast" else 850.0
                    pause_after_word = None
                partials.append(
                    PartialChunk(offset_ms=round(elapsed_ms, 1), text=" ".join(words[:idx]))
                )
                idx += chunk_step_words
            final_offset = elapsed_ms + max(1, len(words) - (idx - chunk_step_words)) * ms_per_word
            is_synthetic_rev = synthetic_revision

        trace = TurnTrace(
            conversation_id=conversation_id,
            turn_id=turn_idx,
            family=family,
            pacing=pacing,
            synthetic_revision=is_synthetic_rev,
            history=[dict(msg) for msg in history],
            partials=partials,
            final=PartialChunk(offset_ms=round(final_offset, 1), text=final_text),
            metadata={
                "has_correction": bool(pre_rev),
                "history_turns": len(history) // 2,
            },
        )
        trace.validate()
        traces.append(trace)

        history.append({"role": "user", "content": final_text})
        if "assistant" in item and item["assistant"]:
            history.append({"role": "assistant", "content": str(item["assistant"])})

    return traces


def _expand_scenarios(
    scenarios: Sequence[Dict[str, Any]],
    prefix: str,
    rng: random.Random,
) -> List[TurnTrace]:
    traces: List[TurnTrace] = []
    family_counts: Dict[str, int] = {}
    for conv_counter, scenario in enumerate(scenarios):
        family = str(scenario["family"])
        fam_idx = family_counts.get(family, 0)
        family_counts[family] = fam_idx + 1
        pacing = "fast" if fam_idx % 2 == 0 else "slow"
        conv_id = f"{prefix}_{family}_{pacing}_{conv_counter:03d}"
        traces.extend(
            convert_raw_dialogue_to_traces(
                conversation_id=conv_id,
                turns=scenario["turns"],
                family=family,
                pacing=pacing,
                rng=rng,
            )
        )
    return traces


def build_sample_fdb_v2_dataset(
    output_dir: Union[str, Path],
    seed: int = 42,
    scenarios_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[TurnTrace], List[TurnTrace]]:
    """Build deterministic dev and held-out FDB-v2 style streaming JSONL traces."""
    rng = random.Random(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios = (
        _RAW_SCENARIOS
        if scenarios_path is None
        else _load_raw_scenarios(Path(scenarios_path).resolve())
    )
    dev_traces = _expand_scenarios(scenarios["dev"], prefix="dev", rng=rng)
    held_out_traces = _expand_scenarios(scenarios["held_out"], prefix="held", rng=rng)
    validate_split_isolation(dev_traces, held_out_traces)

    save_traces(dev_traces, out_path / "dev.jsonl")
    save_traces(held_out_traces, out_path / "held_out.jsonl")
    return dev_traces, held_out_traces
