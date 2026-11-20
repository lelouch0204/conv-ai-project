"""Base controller contract, state tracking, and speech-signal helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_REVISION_PATTERNS = (
    re.compile(r"\bwait\b", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bsorry\b", re.IGNORECASE),
    re.compile(r"\bi meant\b", re.IGNORECASE),
    re.compile(r"\bscratch\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bmake it\b", re.IGNORECASE),
    re.compile(r"\bchange that\b", re.IGNORECASE),
    re.compile(r"\bpush it back\b", re.IGNORECASE),
    re.compile(r"\bnever mind\b", re.IGNORECASE),
)

_DANGLING_TAILS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "at",
    "on",
    "in",
    "of",
    "with",
    "from",
    "by",
    "and",
    "or",
    "but",
    "if",
    "that",
    "is",
    "are",
    "was",
    "wait",
    "actually",
    "sorry",
}

_REQUEST_HEADS = {
    "can",
    "could",
    "would",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
    "who",
    "should",
    "do",
    "book",
    "find",
    "set",
    "draft",
    "explain",
    "summarize",
    "give",
    "suggest",
    "help",
    "remember",
    "note",
    "keep",
    "send",
    "schedule",
    "reserve",
    "order",
    "translate",
    "we",
    "our",
}


def detect_revision_cue(prev_text: str, new_text: str) -> bool:
    """Detect whether the latest partial introduces a semantic correction/pivot cue."""
    prev_words = prev_text.split()
    new_words = new_text.split()
    tail_window = " ".join(new_words[max(0, len(prev_words) - 2) :])
    return any(pat.search(tail_window) for pat in _REVISION_PATTERNS)


def estimate_semantic_readiness(text: str) -> float:
    """Lightweight heuristic score in [0, 1] estimating answerable readiness."""
    words = [w.strip(".,?!;:\"'").lower() for w in text.split() if w.strip()]
    if not words:
        return 0.0

    score = 0.15
    if len(words) >= 5:
        score += 0.25
    if len(words) >= 8:
        score += 0.20
    if len(words) >= 11:
        score += 0.10

    if words[0] in _REQUEST_HEADS or any(w in _REQUEST_HEADS for w in words[:3]):
        score += 0.20

    if text.rstrip().endswith(("?", ".")):
        score += 0.15

    last_word = words[-1]
    if last_word in _DANGLING_TAILS:
        score -= 0.35

    # Penalize if an active mid-turn revision cue appeared very recently in the tail
    recent_tail = " ".join(words[-4:])
    if any(pat.search(recent_tail) for pat in _REVISION_PATTERNS):
        score -= 0.30

    return max(0.0, min(1.0, score))


@dataclass
class DraftDecision:
    """Decision returned by a draft scheduling controller."""

    should_draft: bool
    draft_budget_tokens: int
    discard_existing_draft: bool = False
    reason: str = ""


@dataclass
class ConversationState:
    """Per-conversation raw streaming and speculation observations."""

    conversation_id: str
    turn_id: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)
    previous_partial_text: str = ""
    current_partial_text: str = ""
    partial_history: List[str] = field(default_factory=list)
    partial_version: int = 0
    utterance_start_ms: float = 0.0
    last_partial_ms: float = 0.0
    active_draft_tokens: List[Any] = field(default_factory=list)
    active_draft_version: int = -1
    accepted_draft_ema: float = 12.0
    total_drafted_turns: int = 0
    last_draft_trigger_ms: float = -1e9

    def start_new_turn(
        self,
        turn_id: int,
        history: Sequence[Dict[str, str]],
        utterance_start_ms: float,
    ) -> None:
        """Reset per-turn streaming state while preserving cross-turn EMA."""
        self.turn_id = turn_id
        self.history = [dict(m) for m in history]
        self.previous_partial_text = ""
        self.current_partial_text = ""
        self.partial_history = []
        self.partial_version = 0
        self.utterance_start_ms = utterance_start_ms
        self.last_partial_ms = utterance_start_ms
        self.active_draft_tokens = []
        self.active_draft_version = -1
        self.last_draft_trigger_ms = -1e9

    def update_partial(
        self,
        text: str,
        timestamp_ms: float,
    ) -> None:
        """Record a new incoming partial transcript snapshot."""
        self.previous_partial_text = self.current_partial_text
        self.current_partial_text = text
        self.partial_history.append(text)
        self.partial_version += 1
        self.last_partial_ms = timestamp_ms

    def update_accepted_ema(self, accepted_tokens: int, alpha: float = 0.4) -> None:
        """Update exponential moving average of accepted draft length."""
        self.accepted_draft_ema = (1.0 - alpha) * self.accepted_draft_ema + alpha * float(
            accepted_tokens
        )
        self.total_drafted_turns += 1


@dataclass
class ServerState:
    """Global serving load snapshot exposed to controllers."""

    now_ms: float
    p0_queue_len: int
    p0_active_count: int
    p1_queue_len: int
    p1_active_drafts: int
    current_batch_size: int
    max_batch_size: int
    recent_step_ms: float = 0.0


class BaseController:
    """Abstract interface for all draft-scheduling policies."""

    name: str = "base"

    def is_revised(self, conv_state: ConversationState) -> bool:
        """Default naive revision check; override for custom/learned detectors."""
        return detect_revision_cue(
            conv_state.previous_partial_text,
            conv_state.current_partial_text,
        )

    def readiness_score(self, conv_state: ConversationState) -> float:
        """Default naive semantic readiness score; override for custom triggers."""
        return estimate_semantic_readiness(conv_state.current_partial_text)

    def decide(
        self,
        conv_state: ConversationState,
        server_state: ServerState,
    ) -> DraftDecision:
        """Return whether to start/refresh an optional draft and with what token budget."""
        raise NotImplementedError

    def on_turn_verified(
        self,
        conv_state: ConversationState,
        accepted_tokens: int,
        drafted_tokens: int,
    ) -> None:
        """Optional hook called after P0 verification completes."""
        return None
