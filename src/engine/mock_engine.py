"""Calibrated roofline mock LLM engine for CPU development and scheduling tests."""

from __future__ import annotations

import hashlib
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

_ACTION_GERUNDS = {
    "book": "booking",
    "find": "finding",
    "set": "setting",
    "draft": "drafting",
    "explain": "explaining",
    "summarize": "summarizing",
    "give": "giving",
    "suggest": "suggesting",
    "help": "helping with",
    "remember": "noting",
    "note": "noting",
    "keep": "tracking",
    "send": "sending",
    "schedule": "scheduling",
    "reserve": "reserving",
    "order": "ordering",
    "translate": "translating",
    "recommend": "recommending",
}

_STOP_OPENERS = {
    "can",
    "you",
    "could",
    "would",
    "please",
    "what",
    "is",
    "are",
    "which",
    "how",
    "do",
    "should",
    "i",
    "we",
    "also",
    "to",
}


@dataclass
class _MockBackendState:
    target_tokens: List[str]
    cursor: int
    max_new_tokens: int


class CalibratedMockEngine(BaseLLMEngine):
    """Roofline-calibrated CPU mock engine with deterministic intent-sensitive tokens.

    Models:
      - Decode step latency: ``t_dec(B) = t0_ms + c_ms * B + knee_penalty``
      - Parallel prefill/verify latency: ``prefill_base_ms + prefill_per_token_ms * n``
      - Deterministic token generation where complete monotonic prefixes share high
        overlap with the final response, while short fragments or pre-correction
        prefixes diverge early.
    """

    def __init__(
        self,
        t0_ms: float = 8.0,
        c_ms: float = 0.8,
        batch_knee: int = 4,
        knee_slope_ms: float = 1.6,
        prefill_base_ms: float = 9.0,
        prefill_per_token_ms: float = 0.25,
        history_attn_ms_per_tok: float = 0.015,
    ) -> None:
        self.t0_ms = float(t0_ms)
        self.c_ms = float(c_ms)
        self.batch_knee = int(batch_knee)
        self.knee_slope_ms = float(knee_slope_ms)
        self.prefill_base_ms = float(prefill_base_ms)
        self.prefill_per_token_ms = float(prefill_per_token_ms)
        self.history_attn_ms_per_tok = float(history_attn_ms_per_tok)

    def estimate_decode_step_ms(self, batch_size: int) -> float:
        if batch_size <= 0:
            return 0.0
        if batch_size <= self.batch_knee:
            return self.t0_ms + self.c_ms * batch_size
        extra = batch_size - self.batch_knee
        return (
            self.t0_ms
            + self.c_ms * batch_size
            + self.knee_slope_ms * extra
        )

    def estimate_prefill_ms(
        self,
        current_turn_tokens: int,
        history_tokens: int,
        extra_verify_tokens: int = 0,
    ) -> float:
        return (
            self.prefill_base_ms
            + self.prefill_per_token_ms * (current_turn_tokens + extra_verify_tokens)
            + self.history_attn_ms_per_tok * history_tokens
        )

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        return max(1, int(round(len(text.split()) * 1.3)))

    @staticmethod
    def _history_token_count(history: Sequence[Dict[str, str]]) -> int:
        total = 0
        for msg in history:
            total += max(1, int(round(len(str(msg.get("content", "")).split()) * 1.3)))
        return total

    @staticmethod
    def _tokenize_response(sentence_pieces: Sequence[str]) -> List[str]:
        tokens: List[str] = []
        for idx, piece in enumerate(sentence_pieces):
            if idx == 0 or piece in {".", ",", "?", "!"}:
                tokens.append(piece)
            else:
                tokens.append(" " + piece)
        return tokens

    def _synthesize_target_tokens(
        self,
        history: Sequence[Dict[str, str]],
        prompt_text: str,
        max_new_tokens: int,
    ) -> List[str]:
        clean = prompt_text.strip()
        words = [w.strip(".,?!;:") for w in clean.split() if w.strip(".,?!;:")]
        lowered_words = [w.lower() for w in words]

        # Very short fragments do not reveal enough intent yet
        if len(words) < 4:
            pieces = [
                "Could",
                "you",
                "please",
                "finish",
                "sharing",
                "what",
                "you",
                "would",
                "like",
                "help",
                "with",
                "?",
            ]
            return self._tokenize_response(pieces)[:max_new_tokens]

        # Check if a correction/pivot marker is present in the prompt
        split_match = _CORRECTION_SPLIT_RE.search(clean)
        if split_match:
            post_pivot = clean[split_match.end() :].strip(" .,?!;:")
            tail_words = [w.strip(".,?!;:") for w in post_pivot.split() if w.strip(".,?!;:")]
            summary_tail = tail_words[:10] if tail_words else lowered_words[-6:]
            first_sentence = [
                "Got",
                "it",
                ",",
                "I",
                "have",
                "updated",
                "the",
                "request",
                "to",
                "reflect",
            ] + summary_tail + ["."]
        else:
            # Extract steady action + topic phrase
            content_words: List[str] = []
            for idx, lw in enumerate(lowered_words):
                if idx < 3 and lw in _STOP_OPENERS:
                    continue
                if not content_words and lw in _ACTION_GERUNDS:
                    content_words.append(_ACTION_GERUNDS[lw])
                else:
                    content_words.append(words[idx])
            if not content_words:
                content_words = words[:8]
            topic_slice = content_words[:12]
            first_sentence = [
                "Sure",
                ",",
                "I",
                "can",
                "help",
                "you",
                "with",
            ] + topic_slice + ["."]

        # Deterministic second sentence conditioned on full prompt + history
        digest_src = "|".join(
            [f"{m.get('role')}:{m.get('content')}" for m in history] + [clean]
        )
        digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()
        follow_variants = [
            ["Let", "me", "confirm", "those", "details", "right", "now", "."],
            ["Everything", "is", "lined", "up", "as", "requested", "."],
            ["Please", "let", "me", "know", "if", "you", "need", "changes", "."],
            ["I", "will", "keep", "that", "noted", "for", "your", "session", "."],
        ]
        variant_idx = int(digest[:4], 16) % len(follow_variants)
        second_sentence = follow_variants[variant_idx]

        tokens = self._tokenize_response(first_sentence + second_sentence)
        return tokens[:max_new_tokens]

    def prefill_prompt(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        text: str,
        is_foreground: bool,
        max_new_tokens: int = 48,
    ) -> Tuple[ActiveSequenceState, float]:
        target_tokens = self._synthesize_target_tokens(history, text, max_new_tokens)
        cur_toks = self._estimate_token_count(text)
        hist_toks = self._history_token_count(history)
        prefill_ms = self.estimate_prefill_ms(cur_toks, hist_toks, extra_verify_tokens=0)

        initial_tokens = [target_tokens[0]] if target_tokens else []
        finished = len(target_tokens) <= 1 or max_new_tokens <= 1
        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=is_foreground,
            prompt_tokens_len=cur_toks + hist_toks,
            generated_tokens=initial_tokens,
            backend_state=_MockBackendState(
                target_tokens=target_tokens,
                cursor=len(initial_tokens),
                max_new_tokens=max_new_tokens,
            ),
            finished=finished,
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
        target_tokens = self._synthesize_target_tokens(history, final_text, max_new_tokens)
        accepted, bonus, accepted_count = verify_draft_prefix(draft_tokens, target_tokens)

        cur_toks = self._estimate_token_count(final_text)
        hist_toks = self._history_token_count(history)
        verify_ms = self.estimate_prefill_ms(
            cur_toks, hist_toks, extra_verify_tokens=len(draft_tokens)
        )

        validated = list(accepted)
        if bonus is not None and len(validated) < max_new_tokens:
            validated.append(bonus)

        finished = len(validated) >= len(target_tokens) or len(validated) >= max_new_tokens
        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=True,
            prompt_tokens_len=cur_toks + hist_toks,
            generated_tokens=validated,
            backend_state=_MockBackendState(
                target_tokens=target_tokens,
                cursor=len(validated),
                max_new_tokens=max_new_tokens,
            ),
            finished=finished,
        )
        ver_res = VerificationResult(
            accepted_tokens=accepted,
            bonus_token=bonus,
            accepted_count=accepted_count,
            drafted_count=len(draft_tokens),
            prefill_time_ms=verify_ms,
        )
        return ver_res, state

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
            if seq.finished or backend.cursor >= min(
                len(backend.target_tokens), backend.max_new_tokens
            ):
                seq.finished = True
                emitted.append(None)
                continue
            tok = backend.target_tokens[backend.cursor]
            backend.cursor += 1
            seq.generated_tokens.append(tok)
            if backend.cursor >= min(len(backend.target_tokens), backend.max_new_tokens):
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
        return self._synthesize_target_tokens(history, final_text, max_new_tokens)
