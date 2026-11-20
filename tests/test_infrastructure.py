"""Unit and integration tests for the shared early-drafting infrastructure."""

from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers import (
    BaseController,
    ConversationState,
    DraftDecision,
    FixedIntervalController,
    NoDraftController,
    ServerState,
    make_controller,
)
from src.data import (
    PartialChunk,
    TurnTrace,
    build_sample_fdb_v2_dataset,
    load_traces,
    validate_split_isolation,
)
from src.engine import (
    CalibratedMockEngine,
    HuggingFaceGPUEngine,
    find_first_sentence_token_index,
    verify_draft_prefix,
)
from src.scheduler import PriorityGateway
from src.simulator import (
    DEFAULT_LOAD_RATES_RPS,
    ReplayHarness,
    generate_workload_schedule,
)


class TestInfrastructure(unittest.TestCase):
    def test_dataset_generation_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dev, held = build_sample_fdb_v2_dataset(tmp, seed=42)
            self.assertGreater(len(dev), 10)
            self.assertGreater(len(held), 10)

            reloaded_dev = load_traces(Path(tmp) / "dev.jsonl")
            reloaded_held = load_traces(Path(tmp) / "held_out.jsonl")
            self.assertEqual(len(dev), len(reloaded_dev))
            self.assertEqual(len(held), len(reloaded_held))
            validate_split_isolation(reloaded_dev, reloaded_held)

            # Injecting a duplicate conversation ID across splits must raise
            leaked_conv = TurnTrace(
                conversation_id=reloaded_dev[0].conversation_id,
                turn_id=99,
                family="daily",
                pacing="fast",
                synthetic_revision=False,
                history=[],
                partials=[PartialChunk(100.0, "Completely novel prefix text here")],
                final=PartialChunk(300.0, "Completely novel final text here."),
            )
            with self.assertRaises(ValueError):
                validate_split_isolation(reloaded_dev, reloaded_held + [leaked_conv])

            # Injecting duplicate utterance text across splits must raise
            leaked_text = TurnTrace(
                conversation_id="unique_held_conv_999",
                turn_id=0,
                family="daily",
                pacing="fast",
                synthetic_revision=False,
                history=[],
                partials=[PartialChunk(100.0, "Some unique partial")],
                final=PartialChunk(300.0, reloaded_dev[0].final.text),
            )
            with self.assertRaises(ValueError):
                validate_split_isolation(reloaded_dev, reloaded_held + [leaked_text])

    def test_exact_greedy_verification_cases(self) -> None:
        target = ["Sure", ",", " I", " can", " help", "."]
        self.assertEqual(verify_draft_prefix([], target), ([], "Sure", 0))
        self.assertEqual(verify_draft_prefix(["No", ","], target), ([], "Sure", 0))
        self.assertEqual(
            verify_draft_prefix(["Sure", ",", " wrong"], target),
            (["Sure", ","], " I", 2),
        )
        self.assertEqual(
            verify_draft_prefix(["Sure", ",", " I", " can"], target),
            (["Sure", ",", " I", " can"], " help", 4),
        )
        self.assertEqual(
            verify_draft_prefix(target + [" extra"], target),
            (target, None, len(target)),
        )

        engine = CalibratedMockEngine()
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."}]
        final_text = "Can you recommend a quiet coffee shop near campus with reliable WiFi?"
        ref = engine.generate_greedy_reference(history, final_text, max_new_tokens=48)
        ver, seq = engine.verify_and_prefill_final(
            "req1", history, final_text, draft_tokens=ref[:6], max_new_tokens=48
        )
        self.assertEqual(ver.accepted_count, 6)
        self.assertEqual(ver.bonus_token, ref[6])
        while not seq.finished:
            engine.decode_step([seq])
        self.assertEqual(seq.generated_tokens, ref)

        sent_idx = find_first_sentence_token_index(ref, engine.tokens_to_text)
        self.assertIsNotNone(sent_idx)

    def test_priority_preemption_and_draft_cancellation(self) -> None:
        engine = CalibratedMockEngine()
        controller = FixedIntervalController(interval_ms=0.0, budget_tokens=16, min_words=2)
        gateway = PriorityGateway(engine=engine, controller=controller, max_batch_size=2)

        # Partial 1 triggers a P1 draft
        gateway.on_partial_arrive(
            "conv_a", 0, [], "Book a table for two", utterance_start_ms=0.0, now_ms=200.0
        )
        self.assertEqual(len(gateway.p1_waiting_prefill), 1)
        gateway.step(200.0)  # prefill P1
        gateway.step(220.0)  # decode 1 token of P1
        first_draft_len = len(gateway.conversations["conv_a"].active_draft_tokens)
        self.assertGreaterEqual(first_draft_len, 2)

        # Partial 2 cancels the previous draft tokens and queues exactly 1 replacement draft
        gateway.on_partial_arrive(
            "conv_a", 0, [], "Book a table for two at seven", utterance_start_ms=0.0, now_ms=400.0
        )
        self.assertEqual(len(gateway.p1_waiting_prefill), 1)
        self.assertEqual(len(gateway.p1_decoding), 0)
        self.assertEqual(gateway._turn_canceled_counts[("conv_a", 0)], first_draft_len)

        # Final request on conv_b must preempt P1 prefill on conv_a
        gateway.on_final_arrive(
            "conv_b", 0, [], "Summarize distributed caching.", utterance_start_ms=0.0, utterance_end_ms=410.0
        )
        gateway.step(410.0)
        # conv_b P0 entered decoding while conv_a P1 is still waiting
        self.assertEqual(len(gateway.p0_decoding), 1)
        self.assertEqual(gateway.p0_decoding[0].conversation_id, "conv_b")
        self.assertEqual(len(gateway.p1_waiting_prefill), 1)

        # Gateway discards stale draft on revision cue when controller declines to re-draft
        rev_gw = PriorityGateway(
            engine=engine,
            controller=FixedIntervalController(interval_ms=99999.0, budget_tokens=16, min_words=3),
            max_batch_size=4,
        )
        rev_gw.on_partial_arrive(
            "conv_c", 0, [], "Book a conference room for four", 0.0, 200.0
        )
        rev_gw.step(200.0)
        self.assertGreater(len(rev_gw.conversations["conv_c"].active_draft_tokens), 0)
        rev_gw.on_partial_arrive(
            "conv_c", 0, [], "Book a conference room for four wait no six", 0.0, 700.0
        )
        self.assertEqual(len(rev_gw.conversations["conv_c"].active_draft_tokens), 0)
        self.assertEqual(len(rev_gw.p1_waiting_prefill), 0)

    def test_all_policies_greedy_equivalence_and_paired_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dev_traces, _ = build_sample_fdb_v2_dataset(tmp, seed=7)
        schedule = generate_workload_schedule(
            dev_traces, arrival_rate_rps=1.2, seed=11, target_turns=24
        )
        engine = CalibratedMockEngine()
        harness = ReplayHarness(engine=engine, max_batch_size=8)

        policy_names = [
            "no_draft",
            "fixed_interval",
        ]
        baseline_outputs = None
        for name in policy_names:
            metrics, summary = harness.run(
                schedule=schedule,
                controller=make_controller(name),
                load_regime="intermediate",
                arrival_rate_rps=1.2,
                seed=11,
            )
            self.assertEqual(summary.num_turns, 24)
            outputs = {(m.conversation_id, m.turn_id): m.response_text for m in metrics}
            if baseline_outputs is None:
                baseline_outputs = outputs
            else:
                self.assertEqual(outputs, baseline_outputs, f"{name} outputs diverged from no_draft")

    def test_light_vs_heavy_load_trends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dev_traces, _ = build_sample_fdb_v2_dataset(tmp, seed=42)
        engine = CalibratedMockEngine()
        harness = ReplayHarness(engine=engine, slo_target_ms=190.0, max_batch_size=8)

        # Light load: FixedInterval drafting beats NoDraft on p50 TTFS
        light_sched = generate_workload_schedule(
            dev_traces, arrival_rate_rps=DEFAULT_LOAD_RATES_RPS["light"], seed=1, target_turns=36
        )
        _, no_draft_light = harness.run(light_sched, NoDraftController(), "light", 0.25, 1)
        _, fixed_light = harness.run(light_sched, FixedIntervalController(), "light", 0.25, 1)
        self.assertLess(fixed_light.ttfs_p50_ms, no_draft_light.ttfs_p50_ms)

        # Heavy load: unthrottled FixedInterval incurs higher p95 TTFS and queue wait than NoDraft
        heavy_sched = generate_workload_schedule(
            dev_traces, arrival_rate_rps=DEFAULT_LOAD_RATES_RPS["heavy"], seed=1, target_turns=48
        )
        _, no_draft_heavy = harness.run(heavy_sched, NoDraftController(), "heavy", 4.2, 1)
        _, fixed_heavy = harness.run(heavy_sched, FixedIntervalController(), "heavy", 4.2, 1)
        self.assertGreater(fixed_heavy.ttfs_p95_ms, no_draft_heavy.ttfs_p95_ms)
        self.assertGreater(fixed_heavy.mean_p0_queue_wait_ms, no_draft_heavy.mean_p0_queue_wait_ms)

    def test_warmup_turns_tagging_and_summary_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dev_traces, _ = build_sample_fdb_v2_dataset(tmp, seed=42)
        sched = generate_workload_schedule(
            dev_traces, arrival_rate_rps=1.0, seed=3, target_turns=10
        )
        engine = CalibratedMockEngine()
        harness = ReplayHarness(engine=engine, warmup_turns=4)
        metrics, summary = harness.run(sched, NoDraftController(), "intermediate", 1.0, 3)

        self.assertEqual(len(metrics), 10)
        self.assertEqual(sum(1 for m in metrics if m.is_warmup), 4)
        self.assertTrue(all(m.is_warmup for m in metrics[:4]))
        self.assertTrue(all(not m.is_warmup for m in metrics[4:]))
        self.assertEqual(summary.warmup_turns, 4)
        self.assertEqual(summary.num_turns, 6)

        # Fallback when total completed turns <= warmup_turns
        short_sched = generate_workload_schedule(
            dev_traces, arrival_rate_rps=1.0, seed=3, target_turns=3
        )
        short_metrics, short_summary = harness.run(
            short_sched, NoDraftController(), "intermediate", 1.0, 3
        )
        self.assertEqual(len(short_metrics), 3)
        self.assertTrue(all(m.is_warmup for m in short_metrics))
        self.assertEqual(short_summary.warmup_turns, 3)
        self.assertEqual(short_summary.num_turns, 3)

    def test_controller_revision_and_readiness_hooks(self) -> None:
        conv = ConversationState(conversation_id="conv_hook")
        conv.update_partial("Book a room", 100.0)
        conv.update_partial("Book a room for six people tomorrow", 300.0)
        self.assertEqual(conv.previous_partial_text, "Book a room")
        self.assertEqual(conv.current_partial_text, "Book a room for six people tomorrow")
        self.assertEqual(
            conv.partial_history,
            ["Book a room", "Book a room for six people tomorrow"],
        )

        class _CustomHookController(BaseController):
            name = "custom_hook"

            def is_revised(self, conv_state: ConversationState) -> bool:
                return "RETRACT" in conv_state.current_partial_text

            def decide(
                self,
                conv_state: ConversationState,
                server_state: ServerState,
            ) -> DraftDecision:
                if conv_state.partial_version == 1:
                    return DraftDecision(True, 8, reason="draft_first")
                return DraftDecision(
                    False,
                    0,
                    discard_existing_draft=self.is_revised(conv_state),
                    reason="hold",
                )

        ctrl = _CustomHookController()
        self.assertGreater(ctrl.readiness_score(conv), 0.5)
        self.assertFalse(ctrl.is_revised(conv))

        gw = PriorityGateway(engine=CalibratedMockEngine(), controller=ctrl, max_batch_size=4)
        gw.on_partial_arrive("conv_h", 0, [], "Book a flight to Boston", 0.0, 100.0)
        gw.step(100.0)
        self.assertGreater(len(gw.conversations["conv_h"].active_draft_tokens), 0)

        # Non-revised update preserves candidate draft tokens
        gw.on_partial_arrive("conv_h", 0, [], "Book a flight to Boston tomorrow", 0.0, 300.0)
        self.assertGreater(len(gw.conversations["conv_h"].active_draft_tokens), 0)

        # Custom is_revised() triggers discard_existing_draft=True
        gw.on_partial_arrive("conv_h", 0, [], "Book a flight RETRACT Miami", 0.0, 500.0)
        self.assertEqual(len(gw.conversations["conv_h"].active_draft_tokens), 0)

    def test_hf_engine_batched_decode_prefix_reuse_and_warmup(self) -> None:
        class _StubTensor:
            def __init__(self, arr: np.ndarray) -> None:
                self.arr = np.asarray(arr)

            @property
            def shape(self):
                return self.arr.shape

            @property
            def dtype(self):
                return self.arr.dtype

            @property
            def device(self):
                return "cuda:0"

            def __getitem__(self, item):
                out = self.arr[item]
                return _StubTensor(out) if isinstance(out, np.ndarray) else out

            def __setitem__(self, key, value) -> None:
                val = value.arr if isinstance(value, _StubTensor) else value
                self.arr[key] = val

            def clone(self):
                return _StubTensor(self.arr.copy())

            def contiguous(self):
                return self

            def unsqueeze(self, dim: int):
                return _StubTensor(np.expand_dims(self.arr, axis=dim))

            def argmax(self, dim: int = -1):
                return _StubTensor(np.argmax(self.arr, axis=dim))

            def item(self):
                return self.arr.item()

            def tolist(self):
                return self.arr.tolist()

        class _StubDynamicCache:
            def __init__(self) -> None:
                self.key_cache = []
                self.value_cache = []

            def update(self, k, v, layer_idx: int) -> None:
                if layer_idx < len(self.key_cache):
                    self.key_cache[layer_idx] = _StubTensor(
                        np.concatenate([self.key_cache[layer_idx].arr, k.arr], axis=2)
                    )
                    self.value_cache[layer_idx] = _StubTensor(
                        np.concatenate([self.value_cache[layer_idx].arr, v.arr], axis=2)
                    )
                else:
                    self.key_cache.append(k)
                    self.value_cache.append(v)

        class _StubCudaEvent:
            def __init__(self, enable_timing: bool = True) -> None:
                self.t = 0.0

            def record(self) -> None:
                self.t = 1.0

            def elapsed_time(self, other) -> float:
                return 3.25

        class _StubTorch:
            long = np.int64

            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return True

                @staticmethod
                def synchronize() -> None:
                    return None

                Event = _StubCudaEvent

            @staticmethod
            def no_grad():
                return contextlib.nullcontext()

            @staticmethod
            def tensor(data, dtype=None, device=None):
                return _StubTensor(np.array(data, dtype=dtype or np.int64))

            @staticmethod
            def ones(shape, dtype=None, device=None):
                return _StubTensor(np.ones(shape, dtype=dtype or np.int64))

            @staticmethod
            def zeros(shape, dtype=None, device=None):
                return _StubTensor(np.zeros(shape, dtype=dtype or np.float32))

            @staticmethod
            def arange(start, end, dtype=None, device=None):
                return _StubTensor(np.arange(start, end, dtype=dtype or np.int64))

        class _StubTokenizer:
            eos_token_id = 0
            pad_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = True):
                return [((sum(ord(c) for c in w) % 20) + 5) for w in text.split()]

            def decode(self, ids, skip_special_tokens: bool = True) -> str:
                return " ".join(str(i) for i in ids)

        class _StubModel:
            def __init__(self) -> None:
                self.calls = []
                self.vocab_size = 64
                self.num_heads = 2
                self.head_dim = 4

            def __call__(
                self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache: bool = True,
            ):
                bsz, seq_len = input_ids.shape
                past_len = (
                    past_key_values.key_cache[0].shape[2]
                    if past_key_values is not None and past_key_values.key_cache
                    else 0
                )
                self.calls.append((bsz, seq_len, past_len))

                cache = _StubDynamicCache()
                if past_key_values is not None and past_key_values.key_cache:
                    cache.key_cache = [k.clone() for k in past_key_values.key_cache]
                    cache.value_cache = [v.clone() for v in past_key_values.value_cache]

                k_new = np.zeros((bsz, self.num_heads, seq_len, self.head_dim), dtype=np.float32)
                v_new = np.zeros((bsz, self.num_heads, seq_len, self.head_dim), dtype=np.float32)
                logits = np.zeros((bsz, seq_len, self.vocab_size), dtype=np.float32)

                for b in range(bsz):
                    for s in range(seq_len):
                        pos = (
                            int(position_ids[b, s])
                            if position_ids is not None
                            else past_len + s
                        )
                        tok = int(input_ids[b, s])
                        k_new[b, :, s, :] = float(tok + pos)
                        v_new[b, :, s, :] = float(tok * 2 + pos)

                cache.update(_StubTensor(k_new), _StubTensor(v_new), 0)
                k_all = cache.key_cache[0].arr
                for b in range(bsz):
                    for s in range(seq_len):
                        pos = (
                            int(position_ids[b, s])
                            if position_ids is not None
                            else past_len + s
                        )
                        tok = int(input_ids[b, s])
                        upto = past_len + s + 1
                        k_slice = k_all[b, 0, :upto, 0]
                        if attention_mask is not None:
                            m_slice = attention_mask.arr[b, :upto]
                            attn_sum = int(np.sum(k_slice * m_slice))
                        else:
                            attn_sum = int(np.sum(k_slice))
                        pred = ((tok * 3 + pos * 5 + attn_sum) % 30) + 10
                        logits[b, s, pred] = 10.0

                return SimpleNamespace(logits=_StubTensor(logits), past_key_values=cache)

        stub_model = _StubModel()
        engine = HuggingFaceGPUEngine(
            device="cuda",
            model=stub_model,
            tokenizer=_StubTokenizer(),
            torch_module=_StubTorch(),
        )
        engine._dynamic_cache_cls = _StubDynamicCache

        # 1. Warmup runs without error and leaves prefix cache clean
        engine.warmup(num_passes=1)
        self.assertEqual(len(engine._prefix_cache), 0)
        stub_model.calls.clear()

        # 2. Prefix KV-cache reuse skips recomputing shared tokens
        seq_p1, t_p1 = engine.prefill_prompt("conv1:t0:v1", [], "book table for", True)
        self.assertEqual(t_p1, 3.25)  # CUDA event timing used
        full_len_1 = seq_p1.prompt_tokens_len
        self.assertEqual(stub_model.calls[-1], (1, full_len_1, 0))

        seq_p2, _ = engine.prefill_prompt(
            "conv1:t0:v2", [], "book table for two at seven", True
        )
        # Reuses all tokens of partial 1 (min(lcp, full_len_2 - 1))
        bsz2, suffix_len2, past_len2 = stub_model.calls[-1]
        self.assertEqual(bsz2, 1)
        self.assertGreater(past_len2, 0)
        self.assertEqual(past_len2 + suffix_len2, seq_p2.prompt_tokens_len)

        # 3. Verify matches uncached full-prompt prediction
        ver_res, seq_final = engine.verify_and_prefill_final(
            "conv1:t0:final",
            [],
            "book table for two at seven",
            draft_tokens=seq_p2.generated_tokens,
        )
        self.assertEqual(ver_res.accepted_count, 1)
        self.assertIsNotNone(ver_res.bonus_token)

        # 4. Batched decode across 2 sequences of unequal lengths uses 1 single forward pass
        seq_other, _ = engine.prefill_prompt(
            "conv2:t0:v1", [], "hi", False
        )
        self.assertNotEqual(
            seq_final.backend_state.total_seq_len,
            seq_other.backend_state.total_seq_len,
        )
        stub_model.calls.clear()
        emitted, batch_ms = engine.decode_step([seq_final, seq_other])
        self.assertEqual(batch_ms, 3.25)
        self.assertEqual(len(stub_model.calls), 1)
        self.assertEqual(stub_model.calls[0][0], 2)  # Single batched forward call (B=2)
        self.assertIsNotNone(emitted[0])
        self.assertIsNotNone(emitted[1])
        # Unpadded per-sequence KV caches stay at exact expected lengths
        self.assertEqual(
            seq_final.backend_state.past_key_values.key_cache[0].shape[2],
            seq_final.backend_state.total_seq_len,
        )
        self.assertEqual(
            seq_other.backend_state.past_key_values.key_cache[0].shape[2],
            seq_other.backend_state.total_seq_len,
        )


if __name__ == "__main__":
    unittest.main()
