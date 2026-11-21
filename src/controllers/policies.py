"""Partial-arrival draft policies, including load/reuse control and its ablation."""

from __future__ import annotations

from typing import Any, Dict

from src.controllers.base import (
    BaseController,
    ConversationState,
    DraftDecision,
    ServerState,
)


class NoDraftController(BaseController):
    """Baseline 1: Never drafts during speech; waits until utterance_end."""

    name = "no_draft"

    def decide(
        self,
        conv_state: ConversationState,
        server_state: ServerState,
    ) -> DraftDecision:
        return DraftDecision(should_draft=False, draft_budget_tokens=0, reason="no_draft")


class FixedIntervalController(BaseController):
    """Load/speech-blind baseline: cooldown between partial-triggered drafts.

    This is evaluated on transcript arrivals, not a periodic timer during silence.
    """

    name = "fixed_interval"

    def __init__(
        self,
        interval_ms: float = 500.0,
        budget_tokens: int = 20,
        min_words: int = 3,
    ) -> None:
        self.interval_ms = float(interval_ms)
        self.budget_tokens = int(budget_tokens)
        self.min_words = int(min_words)
        if self.interval_ms < 0 or self.budget_tokens < 1 or self.min_words < 1:
            raise ValueError("interval_ms must be nonnegative; budgets/min_words must be positive")

    def decide(
        self,
        conv_state: ConversationState,
        server_state: ServerState,
    ) -> DraftDecision:
        if len(conv_state.current_partial_text.split()) < self.min_words:
            return DraftDecision(False, 0, reason="too_short")
        elapsed = server_state.now_ms - conv_state.last_draft_trigger_ms
        if conv_state.last_draft_trigger_ms < 0 or elapsed >= self.interval_ms:
            return DraftDecision(True, self.budget_tokens, reason="fixed_interval")
        return DraftDecision(False, 0, reason="interval_cooldown")


class ReuseOnlyController(FixedIntervalController):
    """Ablation: size drafts using past accepted lengths, without load gating."""

    name = "reuse_only"

    def __init__(
        self,
        interval_ms: float = 500.0,
        budget_tokens: int = 20,
        min_words: int = 3,
        min_budget_tokens: int = 4,
    ) -> None:
        super().__init__(interval_ms, budget_tokens, min_words)
        self.min_budget_tokens = int(min_budget_tokens)
        if not 1 <= self.min_budget_tokens <= self.budget_tokens:
            raise ValueError("min_budget_tokens must be in [1, budget_tokens]")

    def decide(self, conv_state: ConversationState, server_state: ServerState) -> DraftDecision:
        decision = super().decide(conv_state, server_state)
        if not decision.should_draft:
            return decision
        budget = max(self.min_budget_tokens, min(self.budget_tokens, round(conv_state.accepted_draft_ema)))
        return DraftDecision(True, budget, reason="reuse_budget")


class LoadReuseAdaptiveController(ReuseOnlyController):
    """Conservative core controller: draft only while foreground is idle.

    Admission is capped at one queued/active background request globally. Existing
    drafts pause at execution boundaries whenever P0 work appears, then resume
    when foreground drains. The inherited budget uses only observed past reuse.
    """

    name = "load_reuse_adaptive"

    def allow_background_work(self, server_state: ServerState) -> bool:
        return server_state.p0_queue_len == 0 and server_state.p0_active_count == 0

    def decide(self, conv_state: ConversationState, server_state: ServerState) -> DraftDecision:
        decision = super().decide(conv_state, server_state)
        if not decision.should_draft:
            return decision
        if not self.allow_background_work(server_state):
            return DraftDecision(False, 0, reason="foreground_busy")
        if server_state.p1_queue_len + server_state.p1_active_drafts >= 1:
            return DraftDecision(False, 0, reason="background_cap")
        if server_state.current_batch_size >= server_state.max_batch_size:
            return DraftDecision(False, 0, reason="batch_full")
        return DraftDecision(True, decision.draft_budget_tokens, reason="idle_gpu_reuse_budget")


_CONTROLLER_REGISTRY: Dict[str, Any] = {
    "no_draft": NoDraftController,
    "fixed_interval": FixedIntervalController,
    "always_draft": FixedIntervalController,
    "reuse_only": ReuseOnlyController,
    "load_reuse_adaptive": LoadReuseAdaptiveController,
}


def make_controller(name: str, **kwargs: Any) -> BaseController:
    """Instantiate a baseline controller by policy name."""
    key = name.strip().lower()
    if key not in _CONTROLLER_REGISTRY:
        valid = ", ".join(sorted(_CONTROLLER_REGISTRY.keys()))
        raise ValueError(f"Unknown controller {name!r}. Valid names: {valid}")
    cls = _CONTROLLER_REGISTRY[key]
    return cls(**kwargs)
