"""vLLM serving engine using PagedAttention, prefix caching, and prompt_logprobs verification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.engine.base import (
    ActiveSequenceState,
    BaseLLMEngine,
    VerificationResult,
)


@dataclass
class _VLLMBackendState:
    context_ids: List[int]
    max_new_tokens: int


class VLLMEngine(BaseLLMEngine):
    """Single-GPU serving engine backed by vLLM with Automatic Prefix Caching."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        llm: Any = None,
        sampling_params_cls: Any = None,
    ) -> None:
        if llm is None or sampling_params_cls is None:
            from vllm import LLM, SamplingParams

            if sampling_params_cls is None:
                sampling_params_cls = SamplingParams
            if llm is None:
                llm = LLM(
                    model=model_name,
                    dtype=dtype,
                    enable_prefix_caching=True,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                )

        self.llm = llm
        self._sampling_params_cls = sampling_params_cls
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id

    def _format_prompt_ids(
        self,
        history: Sequence[Dict[str, str]],
        text: str,
    ) -> List[int]:
        messages = [dict(m) for m in history] + [{"role": "user", "content": text}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return list(self.tokenizer.encode(rendered, add_special_tokens=False))
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return list(self.tokenizer.encode(joined, add_special_tokens=True))

    def _timed_generate(
        self,
        prompts: Sequence[Dict[str, Any]],
        sampling_params: Any,
    ) -> Tuple[List[Any], float]:
        t0 = time.perf_counter()
        outputs = self.llm.generate(list(prompts), sampling_params, use_tqdm=False)
        return list(outputs), (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _extract_greedy_prompt_token(logprob_entry: Optional[Dict[int, Any]]) -> Optional[int]:
        if not logprob_entry:
            return None
        for tok_id, info in logprob_entry.items():
            if getattr(info, "rank", None) == 1:
                return int(tok_id)
        best_tok = max(
            logprob_entry.items(),
            key=lambda kv: getattr(kv[1], "logprob", kv[1]),
        )[0]
        return int(best_tok)

    def warmup(self, num_passes: int = 2) -> None:
        """Warm up vLLM CUDA graphs, prefix cache, and batched decode path."""
        for idx in range(max(1, int(num_passes))):
            seq_a, _ = self.prefill_prompt(
                f"__warmup_a_{idx}", [], "Warm up the voice assistant serving kernels.", True, 4
            )
            seq_b, _ = self.prefill_prompt(
                f"__warmup_b_{idx}", [], "Second sequence for batched decode warmup.", False, 4
            )
            self.decode_step([seq_a, seq_b])
            self.verify_and_prefill_final(
                f"__warmup_a_{idx}:final",
                [],
                "Warm up the voice assistant serving kernels.",
                seq_a.generated_tokens,
                max_new_tokens=4,
            )

    def prefill_prompt(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        text: str,
        is_foreground: bool,
        max_new_tokens: int = 48,
    ) -> Tuple[ActiveSequenceState, float]:
        prompt_ids = self._format_prompt_ids(history, text)
        params = self._sampling_params_cls(temperature=0.0, max_tokens=1)
        outputs, elapsed_ms = self._timed_generate([{"prompt_token_ids": prompt_ids}], params)

        first_tok = int(outputs[0].outputs[0].token_ids[0])
        hit_eos = first_tok == self.eos_token_id
        generated = [] if hit_eos else [first_tok]
        context_ids = prompt_ids if hit_eos else (prompt_ids + [first_tok])

        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=is_foreground,
            prompt_tokens_len=len(prompt_ids),
            generated_tokens=generated,
            backend_state=_VLLMBackendState(
                context_ids=context_ids,
                max_new_tokens=max_new_tokens,
            ),
            finished=hit_eos or max_new_tokens <= 1,
        )
        return state, elapsed_ms

    def verify_and_prefill_final(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        final_text: str,
        draft_tokens: Sequence[Any],
        max_new_tokens: int = 48,
    ) -> Tuple[VerificationResult, ActiveSequenceState]:
        prompt_ids = self._format_prompt_ids(history, final_text)
        draft_ids = [int(t) for t in draft_tokens]
        full_ids = prompt_ids + draft_ids

        params = self._sampling_params_cls(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )
        outputs, elapsed_ms = self._timed_generate([{"prompt_token_ids": full_ids}], params)
        req_out = outputs[0]

        prompt_len = len(prompt_ids)
        prompt_lps = getattr(req_out, "prompt_logprobs", None) or []
        accepted_count = 0
        mismatched_greedy: Optional[int] = None

        for idx, d_tok in enumerate(draft_ids):
            pos = prompt_len + idx
            lp_entry = prompt_lps[pos] if pos < len(prompt_lps) else None
            greedy_tok = self._extract_greedy_prompt_token(lp_entry)
            if greedy_tok is not None and d_tok == greedy_tok and d_tok != self.eos_token_id:
                accepted_count += 1
            else:
                mismatched_greedy = greedy_tok
                break

        accepted_tokens = draft_ids[:accepted_count]
        if accepted_count < len(draft_ids) and mismatched_greedy is not None:
            bonus_raw = mismatched_greedy
        else:
            bonus_raw = int(req_out.outputs[0].token_ids[0])

        hit_eos = bonus_raw == self.eos_token_id
        bonus_token = None if hit_eos else bonus_raw

        validated = list(accepted_tokens)
        if bonus_token is not None and len(validated) < max_new_tokens:
            validated.append(bonus_token)

        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=True,
            prompt_tokens_len=prompt_len,
            generated_tokens=validated,
            backend_state=_VLLMBackendState(
                context_ids=prompt_ids + validated,
                max_new_tokens=max_new_tokens,
            ),
            finished=hit_eos or len(validated) >= max_new_tokens,
        )
        ver_res = VerificationResult(
            accepted_tokens=accepted_tokens,
            bonus_token=bonus_token,
            accepted_count=accepted_count,
            drafted_count=len(draft_ids),
            prefill_time_ms=elapsed_ms,
        )
        return ver_res, state

    def decode_step(
        self,
        active_sequences: Sequence[ActiveSequenceState],
    ) -> Tuple[List[Optional[Any]], float]:
        if not active_sequences:
            return [], 0.0

        emitted: List[Optional[Any]] = [None] * len(active_sequences)
        live: List[Tuple[int, ActiveSequenceState]] = []
        for idx, seq in enumerate(active_sequences):
            backend: _VLLMBackendState = seq.backend_state
            if seq.finished or len(seq.generated_tokens) >= backend.max_new_tokens:
                seq.finished = True
            else:
                live.append((idx, seq))

        if not live:
            return emitted, 0.0

        prompts = [{"prompt_token_ids": seq.backend_state.context_ids} for _, seq in live]
        params = self._sampling_params_cls(temperature=0.0, max_tokens=1)
        outputs, elapsed_ms = self._timed_generate(prompts, params)

        for (orig_idx, seq), out in zip(live, outputs):
            backend = seq.backend_state
            next_tok = int(out.outputs[0].token_ids[0])
            if next_tok == self.eos_token_id:
                seq.finished = True
                continue
            seq.generated_tokens.append(next_tok)
            backend.context_ids.append(next_tok)
            if len(seq.generated_tokens) >= backend.max_new_tokens:
                seq.finished = True
            emitted[orig_idx] = next_tok

        return emitted, elapsed_ms

    def tokens_to_text(self, tokens: Sequence[Any]) -> str:
        if not tokens:
            return ""
        return self.tokenizer.decode([int(t) for t in tokens], skip_special_tokens=True).strip()

    def generate_greedy_reference(
        self,
        history: Sequence[Dict[str, str]],
        final_text: str,
        max_new_tokens: int = 48,
    ) -> List[Any]:
        prompt_ids = self._format_prompt_ids(history, final_text)
        params = self._sampling_params_cls(temperature=0.0, max_tokens=max_new_tokens)
        outputs, _ = self._timed_generate([{"prompt_token_ids": prompt_ids}], params)
        gen_ids = [int(t) for t in outputs[0].outputs[0].token_ids]
        if self.eos_token_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(self.eos_token_id)]
        return gen_ids
