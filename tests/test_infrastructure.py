"""Unit and integration tests for the shared early-drafting infrastructure."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    VLLMEngine,
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

    def test_vllm_engine_verification_batched_decode_and_warmup(self) -> None:
        class _StubTokenizer:
            eos_token_id = 0
            unk_token_id = -1

            def convert_tokens_to_ids(self, tok: str) -> int:
                return 99 if tok == "<|end|>" else -1

            def encode(self, text: str, add_special_tokens: bool = True):
                return [((sum(ord(c) for c in w) % 20) + 5) for w in text.split()]

            def decode(self, ids, skip_special_tokens: bool = True) -> str:
                return " ".join(str(i) for i in ids)

        class _StubSamplingParams:
            def __init__(self, temperature: float = 0.0, max_tokens: int = 1, prompt_logprobs=None):
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.prompt_logprobs = prompt_logprobs

        class _StubLLM:
            def __init__(self) -> None:
                self._tok = _StubTokenizer()
                self.calls = []

            def get_tokenizer(self):
                return self._tok

            @staticmethod
            def _next_tok(prefix_ids):
                return ((sum(prefix_ids) + len(prefix_ids) * 7) % 25) + 10

            def generate(self, prompts, sampling_params, use_tqdm: bool = False):
                self.calls.append((len(prompts), sampling_params.prompt_logprobs))
                results = []
                for p in prompts:
                    ids = list(p["prompt_token_ids"])
                    prompt_lps = None
                    if sampling_params.prompt_logprobs:
                        prompt_lps = [None] + [
                            {self._next_tok(ids[:i]): SimpleNamespace(rank=1, logprob=-0.1)}
                            for i in range(1, len(ids))
                        ]
                    cur = list(ids)
                    out_toks = []
                    for _ in range(sampling_params.max_tokens):
                        nxt = self._next_tok(cur)
                        out_toks.append(nxt)
                        cur.append(nxt)
                    results.append(
                        SimpleNamespace(
                            outputs=[SimpleNamespace(token_ids=out_toks)],
                            prompt_logprobs=prompt_lps,
                        )
                    )
                return results

        stub_llm = _StubLLM()
        engine = VLLMEngine(llm=stub_llm, sampling_params_cls=_StubSamplingParams)

        engine.warmup(num_passes=1)
        self.assertGreaterEqual(len(stub_llm.calls), 3)
        stub_llm.calls.clear()

        # Draft 3 tokens for prompt, then verify exact match vs partial mismatch
        seq_a, _ = engine.prefill_prompt("conv1:t0:v1", [], "book a table for two", False)
        engine.decode_step([seq_a])
        engine.decode_step([seq_a])
        draft = list(seq_a.generated_tokens)
        self.assertEqual(len(draft), 3)

        ver_full, seq_full = engine.verify_and_prefill_final(
            "conv1:t0:final", [], "book a table for two", draft_tokens=draft
        )
        self.assertEqual(ver_full.accepted_count, 3)
        self.assertIsNotNone(ver_full.bonus_token)
        self.assertEqual(len(seq_full.generated_tokens), 4)

        ver_partial, _ = engine.verify_and_prefill_final(
            "conv1:t0:final2", [], "book a table for two", draft_tokens=[draft[0], 999, draft[2]]
        )
        self.assertEqual(ver_partial.accepted_count, 1)
        self.assertEqual(ver_partial.bonus_token, draft[1])

        # Batched decode runs multiple active sequences in one generate() batch
        seq_b, _ = engine.prefill_prompt("conv2:t0:v1", [], "hello there", True)
        stub_llm.calls.clear()
        emitted, _ = engine.decode_step([seq_full, seq_b])
        self.assertEqual(len(stub_llm.calls), 1)
        self.assertEqual(stub_llm.calls[0][0], 2)
        self.assertIsNotNone(emitted[0])
        self.assertIsNotNone(emitted[1])

        # Extra chat stop token (<|end|> -> 99) terminates sequence cleanly
        self.assertIn(99, engine.stop_token_ids)
        stub_llm._next_tok = lambda prefix_ids: 99
        emitted_stop, _ = engine.decode_step([seq_b])
        self.assertIsNone(emitted_stop[0])
        self.assertTrue(seq_b.finished)


if __name__ == "__main__":
    unittest.main()
