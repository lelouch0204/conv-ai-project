"""HuggingFace PyTorch GPU engine with batched decoding, LCP prefix KV reuse, and CUDA timing."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.engine.base import (
    ActiveSequenceState,
    BaseLLMEngine,
    VerificationResult,
)


@dataclass
class _HFBackendState:
    past_key_values: Any
    next_token_id: int
    total_seq_len: int
    max_new_tokens: int


class HuggingFaceGPUEngine(BaseLLMEngine):
    """Single-GPU serving engine using HuggingFace Transformers.

    Features:
      - Warmup passes (``warmup()``) to prime CUDA kernels and allocator pools.
      - CUDA event timing (``torch.cuda.Event``) with ``perf_counter`` CPU fallback.
      - Per-conversation Longest-Common-Prefix (LCP) KV-cache reuse across
        invariant multi-turn history and unchanged prompt prefixes.
      - True left-padded batched ``decode_step`` sharing one model weight load across
        all active P0 and P1 sequences.

    Dependencies (``torch``, ``transformers``) are imported lazily on instantiation
    (unless injected for CPU unit testing) so the rest of the repository remains
    runnable on CPU-only login nodes.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_prefix_cache_entries: int = 64,
        model: Any = None,
        tokenizer: Any = None,
        torch_module: Any = None,
    ) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
        self.torch = torch_module
        self.device = device
        self.max_prefix_cache_entries = max(1, int(max_prefix_cache_entries))

        self._dynamic_cache_cls: Any = None
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                from transformers import DynamicCache

                self._dynamic_cache_cls = DynamicCache
            except ImportError:
                self._dynamic_cache_cls = None

            torch_dtype = (
                getattr(self.torch, dtype) if isinstance(dtype, str) else dtype
            )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                ).to(device)
                model.eval()

        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = model
        self.eos_token_id = self.tokenizer.eos_token_id
        self._prefix_cache: "OrderedDict[str, Tuple[List[int], Any]]" = OrderedDict()

    def _sync_if_cuda(self) -> None:
        cuda_mod = getattr(self.torch, "cuda", None)
        if (
            cuda_mod is not None
            and str(self.device).startswith("cuda")
            and cuda_mod.is_available()
        ):
            cuda_mod.synchronize()

    def _timed_forward(self, fn: Callable[[], Any]) -> Tuple[Any, float]:
        """Run ``fn()`` under ``no_grad`` using CUDA events when running on CUDA."""
        cuda_mod = getattr(self.torch, "cuda", None)
        use_cuda_events = (
            cuda_mod is not None
            and str(self.device).startswith("cuda")
            and cuda_mod.is_available()
            and hasattr(cuda_mod, "Event")
        )
        if use_cuda_events:
            start_evt = cuda_mod.Event(enable_timing=True)
            end_evt = cuda_mod.Event(enable_timing=True)
            start_evt.record()
            with self.torch.no_grad():
                result = fn()
            end_evt.record()
            cuda_mod.synchronize()
            return result, float(start_evt.elapsed_time(end_evt))

        t0 = time.perf_counter()
        with self.torch.no_grad():
            result = fn()
        self._sync_if_cuda()
        return result, (time.perf_counter() - t0) * 1000.0

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

    @staticmethod
    def _conv_cache_key(request_id: str) -> str:
        return request_id.split(":", 1)[0]

    @staticmethod
    def _extract_kv_layers(past_key_values: Any) -> List[Tuple[Any, Any]]:
        if past_key_values is None:
            return []
        if hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))
        if hasattr(past_key_values, "layers"):
            return [(layer.keys, layer.values) for layer in past_key_values.layers]
        return [(k, v) for k, v in past_key_values]

    def _build_kv_cache(
        self,
        layers: Sequence[Tuple[Any, Any]],
        template_kv: Any = None,
    ) -> Any:
        if template_kv is not None and hasattr(template_kv, "update"):
            cache = type(template_kv)()
            for layer_idx, (k, v) in enumerate(layers):
                cache.update(k, v, layer_idx)
            return cache
        if self._dynamic_cache_cls is not None:
            cache = self._dynamic_cache_cls()
            for layer_idx, (k, v) in enumerate(layers):
                cache.update(k, v, layer_idx)
            return cache
        return tuple(layers)

    def _clone_and_crop_kv(self, past_key_values: Any, keep_len: int) -> Any:
        layers = self._extract_kv_layers(past_key_values)
        cropped = [
            (
                k[:, :, :keep_len, :].clone().contiguous(),
                v[:, :, :keep_len, :].clone().contiguous(),
            )
            for k, v in layers
        ]
        return self._build_kv_cache(cropped, template_kv=past_key_values)

    def _lookup_prefix_cache(
        self,
        conv_key: str,
        prompt_ids: Sequence[int],
    ) -> Tuple[int, Any]:
        """Return ``(reuse_len, cloned_past_kv)`` for the longest invariant prefix."""
        entry = self._prefix_cache.get(conv_key)
        if entry is None or len(prompt_ids) <= 1:
            return 0, None
        self._prefix_cache.move_to_end(conv_key)
        cached_ids, cached_kv = entry
        lcp = 0
        for a, b in zip(cached_ids, prompt_ids):
            if a == b:
                lcp += 1
            else:
                break
        # Keep at least the final prompt token uncached so logits predict target[0].
        reuse_len = min(lcp, len(prompt_ids) - 1)
        if reuse_len <= 0:
            return 0, None
        return reuse_len, self._clone_and_crop_kv(cached_kv, reuse_len)

    def _store_prefix_cache(
        self,
        conv_key: str,
        prompt_ids: Sequence[int],
        past_key_values: Any,
    ) -> None:
        if not prompt_ids or past_key_values is None:
            return
        cloned_kv = self._clone_and_crop_kv(past_key_values, len(prompt_ids))
        self._prefix_cache[conv_key] = (list(prompt_ids), cloned_kv)
        self._prefix_cache.move_to_end(conv_key)
        while len(self._prefix_cache) > self.max_prefix_cache_entries:
            self._prefix_cache.popitem(last=False)

    def _forward_with_prefix(
        self,
        full_ids: Sequence[int],
        reuse_len: int,
        past_kv: Any,
    ) -> Tuple[Any, float]:
        suffix_ids = list(full_ids[reuse_len:])
        input_tensor = self.torch.tensor(
            [suffix_ids], dtype=self.torch.long, device=self.device
        )
        if past_kv is None or reuse_len == 0:
            return self._timed_forward(
                lambda: self.model(input_ids=input_tensor, use_cache=True)
            )
        total_len = reuse_len + len(suffix_ids)
        attention_mask = self.torch.ones(
            (1, total_len), dtype=self.torch.long, device=self.device
        )
        position_ids = self.torch.arange(
            reuse_len, total_len, dtype=self.torch.long, device=self.device
        ).unsqueeze(0)
        return self._timed_forward(
            lambda: self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        )

    def warmup(self, num_passes: int = 2) -> None:
        """Warm up prefill, verification, and batched decode paths."""
        dummy_history: List[Dict[str, str]] = []
        for idx in range(max(1, int(num_passes))):
            seq_a, _ = self.prefill_prompt(
                request_id=f"__warmup_a_{idx}:t0:v1",
                history=dummy_history,
                text="Warm up the shared voice assistant serving kernels.",
                is_foreground=True,
                max_new_tokens=4,
            )
            seq_b, _ = self.prefill_prompt(
                request_id=f"__warmup_b_{idx}:t0:v1",
                history=dummy_history,
                text="Second sequence for batched decode kernel warmup.",
                is_foreground=False,
                max_new_tokens=4,
            )
            self.decode_step([seq_a, seq_b])
            self.verify_and_prefill_final(
                request_id=f"__warmup_a_{idx}:t0:final",
                history=dummy_history,
                final_text="Warm up the shared voice assistant serving kernels.",
                draft_tokens=seq_a.generated_tokens,
                max_new_tokens=4,
            )
        self._prefix_cache.clear()
        self._sync_if_cuda()

    def prefill_prompt(
        self,
        request_id: str,
        history: Sequence[Dict[str, str]],
        text: str,
        is_foreground: bool,
        max_new_tokens: int = 48,
    ) -> Tuple[ActiveSequenceState, float]:
        prompt_ids = self._format_prompt_ids(history, text)
        conv_key = self._conv_cache_key(request_id)
        reuse_len, past_kv = self._lookup_prefix_cache(conv_key, prompt_ids)
        outputs, elapsed_ms = self._forward_with_prefix(prompt_ids, reuse_len, past_kv)

        self._store_prefix_cache(conv_key, prompt_ids, outputs.past_key_values)

        first_tok = int(outputs.logits[0, -1, :].argmax(dim=-1).item())
        finished = first_tok == self.eos_token_id or max_new_tokens <= 1
        generated = [] if first_tok == self.eos_token_id else [first_tok]

        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=is_foreground,
            prompt_tokens_len=len(prompt_ids),
            generated_tokens=generated,
            backend_state=_HFBackendState(
                past_key_values=outputs.past_key_values,
                next_token_id=first_tok,
                total_seq_len=len(prompt_ids),
                max_new_tokens=max_new_tokens,
            ),
            finished=finished,
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

        conv_key = self._conv_cache_key(request_id)
        reuse_len, past_kv = self._lookup_prefix_cache(conv_key, prompt_ids)
        outputs, elapsed_ms = self._forward_with_prefix(full_ids, reuse_len, past_kv)

        prompt_len = len(prompt_ids)
        prompt_tail_len = prompt_len - reuse_len
        # Position prompt_tail_len - 1 predicts target[0], ..., + len(draft_ids) predicts bonus
        pred_logits = outputs.logits[
            0, prompt_tail_len - 1 : prompt_tail_len + len(draft_ids), :
        ]
        greedy_preds = pred_logits.argmax(dim=-1).tolist()

        accepted_count = 0
        for idx, d_tok in enumerate(draft_ids):
            if d_tok == int(greedy_preds[idx]) and d_tok != self.eos_token_id:
                accepted_count += 1
            else:
                break

        accepted_tokens = draft_ids[:accepted_count]
        bonus_raw = int(greedy_preds[accepted_count])
        hit_eos = bonus_raw == self.eos_token_id
        bonus_token = None if hit_eos else bonus_raw

        # Cache prompt KV state for future turns, and crop sequence KV to [prompt + accepted].
        self._store_prefix_cache(conv_key, prompt_ids, outputs.past_key_values)
        keep_kv_len = prompt_len + accepted_count
        kv_cache = self._clone_and_crop_kv(outputs.past_key_values, keep_kv_len)

        validated = list(accepted_tokens)
        if bonus_token is not None and len(validated) < max_new_tokens:
            validated.append(bonus_token)

        finished = hit_eos or len(validated) >= max_new_tokens
        state = ActiveSequenceState(
            request_id=request_id,
            is_foreground=True,
            prompt_tokens_len=prompt_len,
            generated_tokens=validated,
            backend_state=_HFBackendState(
                past_key_values=kv_cache,
                next_token_id=bonus_raw,
                total_seq_len=keep_kv_len,
                max_new_tokens=max_new_tokens,
            ),
            finished=finished,
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
            backend: _HFBackendState = seq.backend_state
            if seq.finished or len(seq.generated_tokens) >= backend.max_new_tokens:
                seq.finished = True
            else:
                live.append((idx, seq))

        if not live:
            return emitted, 0.0

        if len(live) == 1:
            orig_idx, seq = live[0]
            backend = seq.backend_state
            token_tensor = self.torch.tensor(
                [[backend.next_token_id]],
                dtype=self.torch.long,
                device=self.device,
            )
            attention_mask = self.torch.ones(
                (1, backend.total_seq_len + 1),
                dtype=self.torch.long,
                device=self.device,
            )
            position_ids = self.torch.tensor(
                [[backend.total_seq_len]],
                dtype=self.torch.long,
                device=self.device,
            )
            outputs, elapsed_ms = self._timed_forward(
                lambda: self.model(
                    input_ids=token_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=backend.past_key_values,
                    use_cache=True,
                )
            )
            backend.past_key_values = outputs.past_key_values
            backend.total_seq_len += 1
            next_tok = int(outputs.logits[0, -1, :].argmax(dim=-1).item())
            self._apply_decoded_token(seq, backend, next_tok, emitted, orig_idx)
            return emitted, elapsed_ms

        # True batched forward pass across all live P0 + P1 sequences via left-padding.
        batch_size = len(live)
        seq_lens = [seq.backend_state.total_seq_len for _, seq in live]
        max_past_len = max(seq_lens)
        all_layers = [
            self._extract_kv_layers(seq.backend_state.past_key_values)
            for _, seq in live
        ]
        num_layers = len(all_layers[0])

        batched_layers: List[Tuple[Any, Any]] = []
        for layer_idx in range(num_layers):
            k0, v0 = all_layers[0][layer_idx]
            _, num_heads, _, head_dim = k0.shape
            k_batch = self.torch.zeros(
                (batch_size, num_heads, max_past_len, head_dim),
                dtype=k0.dtype,
                device=k0.device,
            )
            v_batch = self.torch.zeros(
                (batch_size, num_heads, max_past_len, head_dim),
                dtype=v0.dtype,
                device=v0.device,
            )
            for b, cur_len in enumerate(seq_lens):
                pad_left = max_past_len - cur_len
                k_b, v_b = all_layers[b][layer_idx]
                k_batch[b : b + 1, :, pad_left:max_past_len, :] = k_b[:, :, :cur_len, :]
                v_batch[b : b + 1, :, pad_left:max_past_len, :] = v_b[:, :, :cur_len, :]
            batched_layers.append((k_batch, v_batch))

        template_kv = live[0][1].backend_state.past_key_values
        batched_kv = self._build_kv_cache(batched_layers, template_kv=template_kv)

        input_ids = self.torch.tensor(
            [[seq.backend_state.next_token_id] for _, seq in live],
            dtype=self.torch.long,
            device=self.device,
        )
        attention_mask = self.torch.zeros(
            (batch_size, max_past_len + 1),
            dtype=self.torch.long,
            device=self.device,
        )
        for b, cur_len in enumerate(seq_lens):
            attention_mask[b, max_past_len - cur_len :] = 1
        position_ids = self.torch.tensor(
            [[cur_len] for cur_len in seq_lens],
            dtype=self.torch.long,
            device=self.device,
        )

        outputs, elapsed_ms = self._timed_forward(
            lambda: self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_kv,
                use_cache=True,
            )
        )

        out_layers = self._extract_kv_layers(outputs.past_key_values)
        for b, (orig_idx, seq) in enumerate(live):
            backend = seq.backend_state
            pad_left = max_past_len - seq_lens[b]
            seq_layers = [
                (
                    k[b : b + 1, :, pad_left : max_past_len + 1, :].clone().contiguous(),
                    v[b : b + 1, :, pad_left : max_past_len + 1, :].clone().contiguous(),
                )
                for k, v in out_layers
            ]
            backend.past_key_values = self._build_kv_cache(
                seq_layers, template_kv=backend.past_key_values
            )
            backend.total_seq_len = seq_lens[b] + 1
            next_tok = int(outputs.logits[b, -1, :].argmax(dim=-1).item())
            self._apply_decoded_token(seq, backend, next_tok, emitted, orig_idx)

        return emitted, elapsed_ms

    def _apply_decoded_token(
        self,
        seq: ActiveSequenceState,
        backend: _HFBackendState,
        next_tok: int,
        emitted: List[Optional[Any]],
        orig_idx: int,
    ) -> None:
        backend.next_token_id = next_tok
        if next_tok == self.eos_token_id:
            seq.finished = True
            emitted[orig_idx] = None
            return
        seq.generated_tokens.append(next_tok)
        if len(seq.generated_tokens) >= backend.max_new_tokens:
            seq.finished = True
        emitted[orig_idx] = next_tok

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
        input_tensor = self.torch.tensor([prompt_ids], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            out_ids = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = out_ids[0, len(prompt_ids) :].tolist()
        if self.eos_token_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(self.eos_token_id)]
        return [int(t) for t in gen_ids]
