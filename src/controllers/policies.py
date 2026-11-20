"""Baseline draft-scheduling policies."""

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
    """Baseline 2: Naive PredGen-style drafting at a fixed time interval."""

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

    def decide(
        self,
        conv_state: ConversationState,
        server_state: ServerState,
    ) -> DraftDecision:
        revised = self.is_revised(conv_state)
        if len(conv_state.current_partial_text.split()) < self.min_words:
            return DraftDecision(
                False, 0, discard_existing_draft=revised, reason="too_short"
            )
        elapsed = server_state.now_ms - conv_state.last_draft_trigger_ms
        if conv_state.last_draft_trigger_ms < 0 or elapsed >= self.interval_ms:
            return DraftDecision(True, self.budget_tokens, reason="fixed_interval")
        return DraftDecision(
            False, 0, discard_existing_draft=revised, reason="interval_cooldown"
        )


_CONTROLLER_REGISTRY: Dict[str, Any] = {
    "no_draft": NoDraftController,
    "fixed_interval": FixedIntervalController,
    "always_draft": FixedIntervalController,
}


def make_controller(name: str, **kwargs: Any) -> BaseController:
    """Instantiate a baseline controller by policy name."""
    key = name.strip().lower()
    if key not in _CONTROLLER_REGISTRY:
        valid = ", ".join(sorted(_CONTROLLER_REGISTRY.keys()))
        raise ValueError(f"Unknown controller {name!r}. Valid names: {valid}")
    cls = _CONTROLLER_REGISTRY[key]
    return cls(**kwargs)
