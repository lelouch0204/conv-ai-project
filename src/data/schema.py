"""Canonical streaming speech trace schema and split-isolation utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Union


def _normalize_text(text: str) -> str:
    """Normalize text for split-leakage checks."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


@dataclass
class PartialChunk:
    """A single timestamped partial or final transcript snapshot."""

    offset_ms: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"offset_ms": float(self.offset_ms), "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartialChunk":
        return cls(offset_ms=float(data["offset_ms"]), text=str(data["text"]))


@dataclass
class TurnTrace:
    """Canonical representation of one streaming spoken user turn."""

    conversation_id: str
    turn_id: int
    family: str
    pacing: str
    synthetic_revision: bool
    history: List[Dict[str, str]]
    partials: List[PartialChunk]
    final: PartialChunk
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate timestamp monotonicity and required fields."""
        if not self.conversation_id:
            raise ValueError("conversation_id must be non-empty.")
        if self.turn_id < 0:
            raise ValueError(f"turn_id must be >= 0, got {self.turn_id}.")
        if self.final.offset_ms < 0:
            raise ValueError("final.offset_ms must be >= 0.")

        prev_t = -1.0
        for idx, partial in enumerate(self.partials):
            if partial.offset_ms < 0:
                raise ValueError(
                    f"Partial {idx} offset_ms must be >= 0, got {partial.offset_ms}."
                )
            if partial.offset_ms < prev_t:
                raise ValueError(
                    f"Non-monotonic partial timestamps in {self.conversation_id} "
                    f"turn {self.turn_id}: {partial.offset_ms} < {prev_t}."
                )
            prev_t = partial.offset_ms

        if self.final.offset_ms < prev_t:
            raise ValueError(
                f"final.offset_ms ({self.final.offset_ms}) precedes last partial ({prev_t}) "
                f"in {self.conversation_id} turn {self.turn_id}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "family": self.family,
            "pacing": self.pacing,
            "synthetic_revision": self.synthetic_revision,
            "history": self.history,
            "partials": [p.to_dict() for p in self.partials],
            "final": self.final.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnTrace":
        trace = cls(
            conversation_id=str(data["conversation_id"]),
            turn_id=int(data["turn_id"]),
            family=str(data.get("family", "daily")),
            pacing=str(data.get("pacing", "normal")),
            synthetic_revision=bool(data.get("synthetic_revision", False)),
            history=list(data.get("history", [])),
            partials=[PartialChunk.from_dict(p) for p in data.get("partials", [])],
            final=PartialChunk.from_dict(data["final"]),
            metadata=dict(data.get("metadata", {})),
        )
        trace.validate()
        return trace


def save_traces(traces: Sequence[TurnTrace], path: Union[str, Path]) -> None:
    """Save TurnTrace objects to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            trace.validate()
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: Union[str, Path]) -> List[TurnTrace]:
    """Load TurnTrace objects from a JSONL file."""
    in_path = Path(path)
    traces: List[TurnTrace] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            traces.append(TurnTrace.from_dict(json.loads(stripped)))
    return traces


def validate_split_isolation(
    dev_traces: Sequence[TurnTrace],
    held_out_traces: Sequence[TurnTrace],
) -> None:
    """Ensure zero conversation or utterance/prefix leakage across dev and held-out."""
    dev_convs: Set[str] = {t.conversation_id for t in dev_traces}
    held_convs: Set[str] = {t.conversation_id for t in held_out_traces}
    shared_convs = dev_convs & held_convs
    if shared_convs:
        raise ValueError(
            f"Conversation leakage between dev and held_out: {sorted(shared_convs)[:5]}"
        )

    dev_texts: Set[str] = set()
    for t in dev_traces:
        dev_texts.add(_normalize_text(t.final.text))
        for p in t.partials:
            norm = _normalize_text(p.text)
            if len(norm.split()) >= 4:
                dev_texts.add(norm)

    for t in held_out_traces:
        held_final = _normalize_text(t.final.text)
        if held_final in dev_texts:
            raise ValueError(
                f"Utterance leakage across splits for {t.conversation_id} turn {t.turn_id}: "
                f"{t.final.text!r}"
            )
        for p in t.partials:
            norm = _normalize_text(p.text)
            if len(norm.split()) >= 4 and norm in dev_texts:
                raise ValueError(
                    f"Partial prefix leakage across splits for {t.conversation_id} "
                    f"turn {t.turn_id}: {p.text!r}"
                )
