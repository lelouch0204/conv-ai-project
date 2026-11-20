"""Compact roofline mock LLM engine for CPU login-node tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.engine.base import (
    ActiveSequenceState,
    BaseLLMEngine,
    VerificationResult,
    verify_draft_prefix,
)

_CORRECTION_SPLIT_RE = re.compile(
    r"\b(?:wait(?:\s+no)?|actually|sorry(?:\s+i\s+meant)?|scratch(?:\s+that)?|"
    r"never\s+mind|change\s+that\s+to|make\s+it|instead)\b",
    re.IGNORECASE,
)


@dataclass
class _MockBackendState:
    target_tokens: List[str]
    cursor: int
    max_new_tokens: int


class CalibratedMockEngine(BaseLLMEngine):
    """Lightweight CPU roofline engine with deterministic prefix/revision tokens."""

    def __init__(
        self,
        t0_ms: float = 8.0,
        c_ms: float = 0.8,
        batch_knee: int = 4,
        knee_slope_ms: float = 1.6,
        prefill_base_ms: float = 9.0,
        prefill_per_token_ms: float = 0.25,
    ) -> None:
        self.t0_ms = float(t0_ms)
        self.c_ms = float(c_ms)
        self.batch_knee = int(batch_knee)
        self.knee_slope_ms = float(knee_slope_ms)
        self.prefill_base_ms = float(prefill_base_ms)
        self.prefill_per_token_ms = float(prefill_per_token_ms)

    def estimate_decode_step_ms(self, batch_size: int) -> float:
        if batch_size <= 0:
            return 0.0
        extra = max(0, batch_size - self.batch_knee)
        return self.t0_ms + self.c_ms * batch_size + self.knee_slope_ms * extra

    def estimate_prefill_ms(self, prompt_tokens: int, extra_tokens: int = 0) -> float:
        return self.prefill_base_ms + self.prefill_per_token_ms * (prompt_tokens + extra_tokens)

    @staticmethod
    def _synthesize_target_tokens(prompt_text: str, max_new_tokens: int) -> List[str]:
        clean = prompt_text.strip()
        words = [w.strip(".,?!;:") for w in clean.split() if w.strip(".,?!;:")]
        if len(words) < 4:
            pieces = ["Could", "you", "finish", "your", "request", "?"]
        elif (match := _CORRECTION_SPLIT_RE.search(clean)) is not None:
            tail = [
                w.strip(".,?!;:")
                for w in clean[match.end() :].split()
                if w.strip(".,?!;:")
            ][:10]
            pieces = ["Got", "it", ",", "updating", "to"] + (tail or words[-5:]) + [
                ".",
                "Let",
                "me",
                "confirm",
                "that",
                ".",
            ]
        else:
            pieces = ["Sure", ",", "helping", "with"] + words[:12] + [
                ".",
                "Let",
                "me",
                "confirm",
                "that",
                ".",
            ]

        tokens = [
            p if (idx == 0 or p in {".", ",", "?", "!"}) else " " + p
            for idx, p in enumerate(pieces)
        ]
        return tokens[:max_new_tokens]

    def prefill_prompt(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        text: str,
        is_foreground: bool,
        max_new_tokens: int = 48,
    ) -> Tuple[ActiveSequenceState, float]:
        target_tokens = self._synthesize_target_tokens(text, max_new_tokens)
        prompt_toks = max(1, int(round(len(text.split()) * 1.3)))
        prefill_ms = self.estimate_prefill_ms(prompt_toks)
        initial = [target_tokens[0]] if target_tokens else []
        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=is_foreground,
            prompt_tokens_len=prompt_toks,
            generated_tokens=initial,
            backend_state=_MockBackendState(target_tokens, len(initial), max_new_tokens),
            finished=len(target_tokens) <= 1 or max_new_tokens <= 1,
        )
        return state, prefill_ms

    def verify_and_prefill_final(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        final_text: str,
        draft_tokens: Sequence[Any],
        max_new_tokens: int = 48,
    ) -> Tuple[VerificationResult, ActiveSequenceState]:
        target_tokens = self._synthesize_target_tokens(final_text, max_new_tokens)
        accepted, bonus, accepted_count = verify_draft_prefix(draft_tokens, target_tokens)
        prompt_toks = max(1, int(round(len(final_text.split()) * 1.3)))
        verify_ms = self.estimate_prefill_ms(prompt_toks, extra_tokens=len(draft_tokens))

        validated = list(accepted)
        if bonus is not None and len(validated) < max_new_tokens:
            validated.append(bonus)

        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=True,
            prompt_tokens_len=prompt_toks,
            generated_tokens=validated,
            backend_state=_MockBackendState(target_tokens, len(validated), max_new_tokens),
            finished=len(validated) >= len(target_tokens) or len(validated) >= max_new_tokens,
        )
        return (
            VerificationResult(accepted, bonus, accepted_count, len(draft_tokens), verify_ms),
            state,
        )

    def decode_step(
        self,
        active_sequences: Sequence[ActiveSequenceState],
    ) -> Tuple[List[Optional[Any]], float]:
        if not active_sequences:
            return [], 0.0
        step_ms = self.estimate_decode_step_ms(len(active_sequences))
        emitted: List[Optional[Any]] = []
        for seq in active_sequences:
            backend: _MockBackendState = seq.backend_state
            limit = min(len(backend.target_tokens), backend.max_new_tokens)
            if seq.finished or backend.cursor >= limit:
                seq.finished = True
                emitted.append(None)
                continue
            tok = backend.target_tokens[backend.cursor]
            backend.cursor += 1
            seq.generated_tokens.append(tok)
            if backend.cursor >= limit:
                seq.finished = True
            emitted.append(tok)
        return emitted, step_ms

    def tokens_to_text(self, tokens: Sequence[Any]) -> str:
        return "".join(str(t) for t in tokens).strip()

    def generate_greedy_reference(
        self,
        history: Sequence[Dict[str, str]],
        final_text: str,
        max_new_tokens: int = 48,
    ) -> List[Any]:
        return self._synthesize_target_tokens(final_text, max_new_tokens)
