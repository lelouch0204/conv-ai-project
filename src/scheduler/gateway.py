"""Shared-GPU Priority Gateway managing Foreground (P0) and Draft (P1) queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from src.controllers.base import (
    BaseController,
    ConversationState,
    ServerState,
)
from src.engine.base import (
    ActiveSequenceState,
    BaseLLMEngine,
    VerificationResult,
    find_first_sentence_token_index,
)


@dataclass
class _ForegroundRequest:
    request_id: str
    conversation_id: str
    turn_id: int
    family: str
    pacing: str
    synthetic_revision: bool
    history: List[Dict[str, str]]
    final_text: str
    utterance_start_ms: float
    utterance_end_ms: float
    draft_tokens: List[Any]
    drafted_tokens_total: int
    canceled_tokens_total: int
    seq_state: Optional[ActiveSequenceState] = None
    queue_wait_ms: float = 0.0
    ttft_ms: Optional[float] = None
    ttfs_ms: Optional[float] = None
    verification_result: Optional[VerificationResult] = None


@dataclass
class _DraftRequest:
    request_id: str
    conversation_id: str
    turn_id: int
    partial_version: int
    history: List[Dict[str, str]]
    partial_text: str
    remaining_budget: int
    seq_state: Optional[ActiveSequenceState] = None
    canceled: bool = False


@dataclass
class TurnCompletionRecord:
    """Raw completion record returned by PriorityGateway when a P0 turn finishes."""

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
    response_tokens: List[Any]
    response_text: str


class PriorityGateway:
    """Gateway enforcing P0 > P1 priority, <=1 active draft/conv, and greedy verification."""

    def __init__(
        self,
        engine: BaseLLMEngine,
        controller: BaseController,
        max_batch_size: int = 8,
        max_new_tokens: int = 48,
        min_sentence_tokens: int = 4,
        sentence_fallback_tokens: int = 24,
    ) -> None:
        self.engine = engine
        self.controller = controller
        self.max_batch_size = int(max_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.min_sentence_tokens = int(min_sentence_tokens)
        self.sentence_fallback_tokens = int(sentence_fallback_tokens)

        self.conversations: Dict[str, ConversationState] = {}
        self._turn_drafted_counts: Dict[Tuple[str, int], int] = {}
        self._turn_canceled_counts: Dict[Tuple[str, int], int] = {}

        self.p0_waiting_prefill: Deque[_ForegroundRequest] = deque()
        self.p0_decoding: List[_ForegroundRequest] = []
        self.p1_waiting_prefill: Deque[_DraftRequest] = deque()
        self.p1_decoding: List[_DraftRequest] = []
        self._recent_step_ms: float = 0.0

    def _get_or_create_conv(self, conv_id: str) -> ConversationState:
        if conv_id not in self.conversations:
            self.conversations[conv_id] = ConversationState(conversation_id=conv_id)
        return self.conversations[conv_id]

    def get_server_state(self, now_ms: float) -> ServerState:
        return ServerState(
            now_ms=now_ms,
            p0_queue_len=len(self.p0_waiting_prefill),
            p0_active_count=len(self.p0_decoding),
            p1_queue_len=len(self.p1_waiting_prefill),
            p1_active_drafts=len(self.p1_decoding),
            current_batch_size=len(self.p0_decoding) + len(self.p1_decoding),
            max_batch_size=self.max_batch_size,
            recent_step_ms=self._recent_step_ms,
        )

    def has_work(self) -> bool:
        return bool(
            self.p0_waiting_prefill
            or self.p0_decoding
            or self.p1_waiting_prefill
            or self.p1_decoding
        )

    def _cancel_draft_for_conversation(
        self,
        conv_id: str,
        turn_id: int,
        discard_generated_tokens: bool,
    ) -> None:
        """Remove any queued or decoding P1 draft for ``conv_id``."""
        conv_state = self.conversations.get(conv_id)
        key = (conv_id, turn_id)

        kept_wait: Deque[_DraftRequest] = deque()
        while self.p1_waiting_prefill:
            req = self.p1_waiting_prefill.popleft()
            if req.conversation_id == conv_id:
                req.canceled = True
            else:
                kept_wait.append(req)
        self.p1_waiting_prefill = kept_wait

        kept_dec: List[_DraftRequest] = []
        for req in self.p1_decoding:
            if req.conversation_id == conv_id:
                req.canceled = True
            else:
                kept_dec.append(req)
        self.p1_decoding = kept_dec

        if discard_generated_tokens and conv_state is not None and conv_state.active_draft_tokens:
            self._turn_canceled_counts[key] = self._turn_canceled_counts.get(key, 0) + len(
                conv_state.active_draft_tokens
            )
            conv_state.active_draft_tokens = []
            conv_state.active_draft_version = -1

    def on_partial_arrive(
        self,
        conversation_id: str,
        turn_id: int,
        history: Sequence[Dict[str, str]],
        partial_text: str,
        utterance_start_ms: float,
        now_ms: float,
    ) -> None:
        """Handle an incoming streaming partial transcript chunk."""
        conv = self._get_or_create_conv(conversation_id)
        if conv.turn_id != turn_id or conv.partial_version == 0:
            conv.start_new_turn(turn_id=turn_id, history=history, utterance_start_ms=utterance_start_ms)
            self._turn_drafted_counts[(conversation_id, turn_id)] = 0
            self._turn_canceled_counts[(conversation_id, turn_id)] = 0

        conv.update_partial(partial_text, timestamp_ms=now_ms)
        server_state = self.get_server_state(now_ms)
        decision = self.controller.decide(conv, server_state)

        if decision.should_draft and decision.draft_budget_tokens > 0:
            # Cancel any previous draft for this conversation before launching the new version
            self._cancel_draft_for_conversation(
                conversation_id, turn_id, discard_generated_tokens=True
            )
            conv.last_draft_trigger_ms = now_ms
            conv.active_draft_version = conv.partial_version
            draft_req = _DraftRequest(
                request_id=f"{conversation_id}:t{turn_id}:v{conv.partial_version}",
                conversation_id=conversation_id,
                turn_id=turn_id,
                partial_version=conv.partial_version,
                history=[dict(m) for m in history],
                partial_text=partial_text,
                remaining_budget=int(decision.draft_budget_tokens),
            )
            self.p1_waiting_prefill.append(draft_req)
        else:
            # Stop extending an obsolete P1 draft; discard tokens if controller signals revision
            self._cancel_draft_for_conversation(
                conversation_id,
                turn_id,
                discard_generated_tokens=bool(decision.discard_existing_draft),
            )

    def on_final_arrive(
        self,
        conversation_id: str,
        turn_id: int,
        history: Sequence[Dict[str, str]],
        final_text: str,
        utterance_start_ms: float,
        utterance_end_ms: float,
        family: str = "daily",
        pacing: str = "fast",
        synthetic_revision: bool = False,
    ) -> None:
        """Handle utterance_end by harvesting any latest draft and queueing P0 work."""
        conv = self._get_or_create_conv(conversation_id)
        if conv.turn_id != turn_id or (conv.partial_version == 0 and not conv.current_partial_text):
            conv.start_new_turn(turn_id=turn_id, history=history, utterance_start_ms=utterance_start_ms)
            self._turn_drafted_counts.setdefault((conversation_id, turn_id), 0)
            self._turn_canceled_counts.setdefault((conversation_id, turn_id), 0)

        # Stop any still-running P1 draft work without deleting candidate tokens already generated
        self._cancel_draft_for_conversation(
            conversation_id, turn_id, discard_generated_tokens=False
        )
        candidate_draft = list(conv.active_draft_tokens)
        conv.active_draft_tokens = []
        conv.active_draft_version = -1

        key = (conversation_id, turn_id)
        drafted_total = self._turn_drafted_counts.get(key, 0)
        canceled_total = self._turn_canceled_counts.get(key, 0)

        fg_req = _ForegroundRequest(
            request_id=f"{conversation_id}:t{turn_id}:final",
            conversation_id=conversation_id,
            turn_id=turn_id,
            family=family,
            pacing=pacing,
            synthetic_revision=synthetic_revision,
            history=[dict(m) for m in history],
            final_text=final_text,
            utterance_start_ms=utterance_start_ms,
            utterance_end_ms=utterance_end_ms,
            draft_tokens=candidate_draft,
            drafted_tokens_total=drafted_total,
            canceled_tokens_total=canceled_total,
        )
        self.p0_waiting_prefill.append(fg_req)

    def _check_sentence_and_ttft(self, req: _ForegroundRequest, now_ms: float) -> None:
        assert req.seq_state is not None
        tokens = req.seq_state.generated_tokens
        if req.ttft_ms is None and len(tokens) >= 1:
            req.ttft_ms = max(0.0, now_ms - req.utterance_end_ms)
        if req.ttfs_ms is None:
            sent_idx = find_first_sentence_token_index(
                tokens,
                self.engine.tokens_to_text,
                min_tokens=self.min_sentence_tokens,
                fallback_tokens=self.sentence_fallback_tokens,
            )
            if sent_idx is not None:
                req.ttfs_ms = max(0.0, now_ms - req.utterance_end_ms)
            elif req.seq_state.finished and tokens:
                req.ttfs_ms = max(0.0, now_ms - req.utterance_end_ms)

    def _finalize_foreground_request(
        self,
        req: _ForegroundRequest,
        completion_ms: float,
    ) -> TurnCompletionRecord:
        assert req.seq_state is not None
        self._check_sentence_and_ttft(req, completion_ms)

        out_tokens = list(req.seq_state.generated_tokens)
        out_text = self.engine.tokens_to_text(out_tokens)

        accepted = req.verification_result.accepted_count if req.verification_result else 0
        latest_draft_len = len(req.draft_tokens)
        rejected_at_verify = max(0, latest_draft_len - accepted)
        total_canceled = req.canceled_tokens_total + rejected_at_verify
        total_drafted = req.drafted_tokens_total

        conv = self.conversations.get(req.conversation_id)
        if conv is not None and total_drafted > 0:
            conv.update_accepted_ema(accepted)
            self.controller.on_turn_verified(conv, accepted, total_drafted)

        ttft_val = req.ttft_ms if req.ttft_ms is not None else max(0.0, completion_ms - req.utterance_end_ms)
        ttfs_val = req.ttfs_ms if req.ttfs_ms is not None else max(0.0, completion_ms - req.utterance_end_ms)
        acc_rate = (accepted / total_drafted) if total_drafted > 0 else 0.0

        return TurnCompletionRecord(
            conversation_id=req.conversation_id,
            turn_id=req.turn_id,
            family=req.family,
            pacing=req.pacing,
            synthetic_revision=req.synthetic_revision,
            utterance_start_ms=req.utterance_start_ms,
            utterance_end_ms=req.utterance_end_ms,
            completion_ms=completion_ms,
            ttft_ms=ttft_val,
            ttfs_ms=ttfs_val,
            total_response_ms=max(0.0, completion_ms - req.utterance_end_ms),
            p0_queue_wait_ms=req.queue_wait_ms,
            drafted_tokens=total_drafted,
            accepted_tokens=accepted,
            canceled_tokens=total_canceled,
            acceptance_rate=acc_rate,
            response_tokens=out_tokens,
            response_text=out_text,
        )

    def _evict_p1_if_needed_for_p0(self) -> None:
        """Evict background P1 drafts when foreground P0 needs batch capacity."""
        while self.p1_decoding and (
            len(self.p0_decoding) + len(self.p1_decoding) >= self.max_batch_size
        ):
            evicted = self.p1_decoding.pop()
            evicted.canceled = True

    def step(self, now_ms: float) -> Tuple[float, List[TurnCompletionRecord]]:
        """Execute one GPU scheduling step and return ``(step_ms, completions)``."""
        completions: List[TurnCompletionRecord] = []

        # 1. Strict Priority 0 prefill/verification when batch capacity is available
        if self.p0_waiting_prefill:
            self._evict_p1_if_needed_for_p0()
            if len(self.p0_decoding) < self.max_batch_size:
                req = self.p0_waiting_prefill.popleft()
                req.queue_wait_ms = max(0.0, now_ms - req.utterance_end_ms)

                if req.draft_tokens:
                    ver_res, seq_state = self.engine.verify_and_prefill_final(
                        request_id=req.request_id,
                        history=req.history,
                        final_text=req.final_text,
                        draft_tokens=req.draft_tokens,
                        max_new_tokens=self.max_new_tokens,
                    )
                    req.verification_result = ver_res
                    req.seq_state = seq_state
                    step_ms = max(0.1, ver_res.prefill_time_ms)
                else:
                    seq_state, prefill_ms = self.engine.prefill_prompt(
                        request_id=req.request_id,
                        history=req.history,
                        text=req.final_text,
                        is_foreground=True,
                        max_new_tokens=self.max_new_tokens,
                    )
                    req.seq_state = seq_state
                    step_ms = max(0.1, prefill_ms)

                end_ms = now_ms + step_ms
                self._recent_step_ms = step_ms
                self._check_sentence_and_ttft(req, end_ms)
                if req.seq_state.finished:
                    completions.append(self._finalize_foreground_request(req, end_ms))
                else:
                    self.p0_decoding.append(req)
                return step_ms, completions

        # 2. If no P0 is waiting for prefill and a P1 draft needs prefill while batch has room
        if (
            not self.p0_waiting_prefill
            and self.p1_waiting_prefill
            and (len(self.p0_decoding) + len(self.p1_decoding)) < self.max_batch_size
        ):
            draft_req = self.p1_waiting_prefill.popleft()
            seq_state, prefill_ms = self.engine.prefill_prompt(
                request_id=draft_req.request_id,
                history=draft_req.history,
                text=draft_req.partial_text,
                is_foreground=False,
                max_new_tokens=max(1, draft_req.remaining_budget),
            )
            draft_req.seq_state = seq_state
            step_ms = max(0.1, prefill_ms)
            self._recent_step_ms = step_ms

            conv = self.conversations.get(draft_req.conversation_id)
            key = (draft_req.conversation_id, draft_req.turn_id)
            emitted_n = len(seq_state.generated_tokens)
            self._turn_drafted_counts[key] = self._turn_drafted_counts.get(key, 0) + emitted_n
            if conv is not None and conv.active_draft_version == draft_req.partial_version:
                conv.active_draft_tokens = list(seq_state.generated_tokens)
            draft_req.remaining_budget -= emitted_n

            if draft_req.remaining_budget > 0 and not seq_state.finished:
                self.p1_decoding.append(draft_req)
            return step_ms, completions

        # 3. Run one batched decode step over active P0 + P1 sequences
        batch_seqs: List[ActiveSequenceState] = []
        p0_active = list(self.p0_decoding)
        for req in p0_active:
            assert req.seq_state is not None
            batch_seqs.append(req.seq_state)

        remaining_slots = max(0, self.max_batch_size - len(batch_seqs))
        p1_active = list(self.p1_decoding[:remaining_slots])
        for dreq in p1_active:
            assert dreq.seq_state is not None
            batch_seqs.append(dreq.seq_state)

        if not batch_seqs:
            return 0.0, completions

        emitted_tokens, step_ms = self.engine.decode_step(batch_seqs)
        step_ms = max(0.1, step_ms)
        self._recent_step_ms = step_ms
        end_ms = now_ms + step_ms

        # Update P0 requests
        still_p0: List[_ForegroundRequest] = []
        for req in p0_active:
            assert req.seq_state is not None
            self._check_sentence_and_ttft(req, end_ms)
            if req.seq_state.finished:
                completions.append(self._finalize_foreground_request(req, end_ms))
            else:
                still_p0.append(req)
        self.p0_decoding = still_p0

        # Update P1 drafts
        p1_active_ids = {id(d) for d in p1_active}
        still_p1: List[_DraftRequest] = []
        for idx, dreq in enumerate(self.p1_decoding):
            if id(dreq) not in p1_active_ids:
                still_p1.append(dreq)
                continue
            assert dreq.seq_state is not None
            tok_idx = len(p0_active) + p1_active.index(dreq)
            new_tok = emitted_tokens[tok_idx]
            if new_tok is not None:
                key = (dreq.conversation_id, dreq.turn_id)
                self._turn_drafted_counts[key] = self._turn_drafted_counts.get(key, 0) + 1
                conv = self.conversations.get(dreq.conversation_id)
                if conv is not None and conv.active_draft_version == dreq.partial_version:
                    conv.active_draft_tokens.append(new_tok)
                dreq.remaining_budget -= 1
            if dreq.remaining_budget > 0 and not dreq.seq_state.finished:
                still_p1.append(dreq)
        self.p1_decoding = still_p1

        return step_ms, completions
