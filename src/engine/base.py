"""Base LLM engine interface and exact greedy draft-verification utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class VerificationResult:
    """Outcome of a single parallel verification pass against the final prompt."""

    accepted_tokens: List[Any]
    bonus_token: Optional[Any]
    accepted_count: int
    drafted_count: int
    prefill_time_ms: float

    @property
    def validated_tokens(self) -> List[Any]:
        tokens = list(self.accepted_tokens)
        if self.bonus_token is not None:
            tokens.append(self.bonus_token)
        return tokens


@dataclass
class ActiveSequenceState:
    """Per-request active decoding state inside an LLM engine."""

    request_id: str
    is_foreground: bool
    prompt_tokens_len: int
    generated_tokens: List[Any] = field(default_factory=list)
    backend_state: Any = None
    finished: bool = False


def verify_draft_prefix(
    draft_tokens: Sequence[Any],
    target_tokens: Sequence[Any],
) -> Tuple[List[Any], Optional[Any], int]:
    """Verify draft tokens against exact greedy target tokens.

    Returns ``(accepted_tokens, bonus_token, accepted_count)`` where
    ``accepted_tokens + ([bonus_token] if bonus_token is not None else [])``
    is strictly equal to ``target_tokens[:accepted_count + (1 if bonus_token is not None else 0)]``.
    """
    max_cmp = min(len(draft_tokens), len(target_tokens))
    accepted_count = 0
    for idx in range(max_cmp):
        if draft_tokens[idx] == target_tokens[idx]:
            accepted_count += 1
        else:
            break

    accepted_tokens = list(target_tokens[:accepted_count])
    bonus_token = (
        target_tokens[accepted_count]
        if accepted_count < len(target_tokens)
        else None
    )
    return accepted_tokens, bonus_token, accepted_count


_NON_SENTENCE_ABBREVS = ("dr.", "mr.", "ms.", "mrs.", "p.m.", "a.m.", "e.g.", "i.e.")


def find_first_sentence_token_index(
    tokens: Sequence[Any],
    tokens_to_text_fn: Callable[[Sequence[Any]], str],
    min_tokens: int = 4,
    fallback_tokens: int = 24,
) -> Optional[int]:
    """Return the 0-based index of the token completing the first TTS sentence."""
    if not tokens:
        return None
    limit = min(len(tokens), fallback_tokens)
    for idx in range(limit):
        if idx + 1 < min_tokens:
            continue
        prefix_text = tokens_to_text_fn(tokens[: idx + 1]).rstrip()
        if prefix_text.endswith((".", "?", "!")):
            lowered = prefix_text.lower()
            if any(lowered.endswith(abbr) for abbr in _NON_SENTENCE_ABBREVS):
                continue
            return idx
    if len(tokens) >= fallback_tokens:
        return fallback_tokens - 1
    return None


class BaseLLMEngine:
    """Abstract serving engine supporting prefill, verification, and batched decoding."""

    def warmup(self, num_passes: int = 2) -> None:
        """Optional kernel/allocator warmup called once before running benchmarks."""
        return None

    def prefill_prompt(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        text: str,
        is_foreground: bool,
        max_new_tokens: int = 48,
    ) -> Tuple[ActiveSequenceState, float]:
        """Prefill a prompt and produce the initial greedy token."""
        raise NotImplementedError

    def verify_and_prefill_final(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        final_text: str,
        draft_tokens: Sequence[Any],
        max_new_tokens: int = 48,
    ) -> Tuple[VerificationResult, ActiveSequenceState]:
        """Verify draft tokens in one parallel pass over final prompt + draft."""
        raise NotImplementedError

    def decode_step(
        self,
        active_sequences: Sequence[ActiveSequenceState],
    ) -> Tuple[List[Optional[Any]], float]:
        """Run one batched decode step across active foreground & background sequences."""
        raise NotImplementedError

    def tokens_to_text(self, tokens: Sequence[Any]) -> str:
        """Convert generated token IDs/pieces into readable text."""
        raise NotImplementedError

    def generate_greedy_reference(
        self,
        history: Sequence[Dict[str, str]],
        final_text: str,
        max_new_tokens: int = 48,
    ) -> List[Any]:
        """Produce the non-speculative greedy baseline token sequence for verification checks."""
        raise NotImplementedError
