"""Regressions for unsafe verification and invalid paired comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers import ConversationState, LoadReuseAdaptiveController, ReuseOnlyController, make_controller
from src.data import load_traces
from src.engine import CalibratedMockEngine, VLLMEngine
from src.scheduler import PriorityGateway
from src.simulator import ReplayHarness, generate_workload_schedule
from src.validation import audit_turn_rows


class StubParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StubTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


class StubLLM:
    """Deterministic causal model with optional malformed scoring responses."""

    def __init__(self):
        self.scoring = "complete"
        self.calls = []
        self.resets = 0

    def get_tokenizer(self):
        return StubTokenizer()

    def reset_prefix_cache(self):
        self.resets += 1
        return True

    @staticmethod
    def next_token(ids):
        return (sum(ids) + 7 * len(ids)) % 25 + 10

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append(params)
        results = []
        for prompt in prompts:
            ids = list(prompt["prompt_token_ids"])
            logs = None
            if getattr(params, "prompt_logprobs", None) and self.scoring != "missing":
                logs = [None] + [
                    {self.next_token(ids[:i]): SimpleNamespace(rank=1 if self.scoring == "complete" else 2, logprob=-.1)}
                    for i in range(1, len(ids))
                ]
            tokens = []
            for _ in range(params.max_tokens):
                token = self.next_token(ids + tokens)
                tokens.append(token)
                if token == 0:
                    break
            results.append(SimpleNamespace(prompt_logprobs=logs, outputs=[SimpleNamespace(token_ids=tokens)]))
        return results


class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.llm = StubLLM()
        self.engine = VLLMEngine(llm=self.llm, sampling_params_cls=StubParams)

    def test_missing_or_incomplete_scores_cannot_release_invalid_tokens(self):
        for scoring in ("missing", "candidate_only"):
            self.llm.scoring = scoring
            with self.subTest(scoring=scoring), self.assertRaisesRegex(RuntimeError, "rank-1"):
                self.engine.verify_and_prefill_final("bad", [], "new request", [999])

    def test_prompt_changes_and_all_mismatch_positions_match_full_reference(self):
        ref = self.engine.generate_greedy_reference([], "final request", 12)
        old = self.engine.generate_greedy_reference([], "obsolete request", 8)
        drafts = [[], old, ref[:5], ref, ref + [999], [999] + ref[1:5]]
        drafts.extend(ref[:i] + [999] + ref[i + 1:8] for i in range(1, 8))
        for draft in drafts:
            with self.subTest(draft=draft):
                result, state = self.engine.verify_and_prefill_final("final", [], "final request", draft, 12)
                while not state.finished:
                    self.engine.decode_step([state])
                self.assertEqual(state.generated_tokens, ref)
                self.assertLessEqual(len(result.validated_tokens), 12)

    def test_output_cap_and_eos_at_mismatch(self):
        ref = self.engine.generate_greedy_reference([], "final", 10)
        result, state = self.engine.verify_and_prefill_final("cap", [], "final", ref, 3)
        self.assertEqual(state.generated_tokens, ref[:3])
        self.assertEqual(result.accepted_count, 3)
        self.assertIsNone(result.bonus_token)
        self.assertTrue(state.finished)
        self.llm.next_token = lambda ids: 0
        result, state = self.engine.verify_and_prefill_final("eos", [], "final", [999], 3)
        self.assertEqual(state.generated_tokens, [])
        self.assertIsNone(result.bonus_token)
        self.assertTrue(state.finished)

    def test_cache_reset_and_sampling_are_explicit(self):
        self.engine.reset_prefix_cache()
        self.assertEqual(self.llm.resets, 1)
        self.engine.prefill_prompt("r", [], "hello", True)
        params = self.llm.calls[-1]
        self.assertEqual((params.temperature, params.repetition_penalty, params.presence_penalty, params.frequency_penalty), (0., 1., 0., 0.))
        self.assertEqual(params.stop_token_ids, [0])
        self.llm.reset_prefix_cache = lambda: False
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            self.engine.reset_prefix_cache()

    def test_load_controller_pauses_existing_draft_and_resumes_after_foreground(self):
        gateway = PriorityGateway(CalibratedMockEngine(), LoadReuseAdaptiveController(), record_trace=True)
        gateway.on_partial_arrive("background", 0, [], "Book a table for two", 0., 100.)
        now = 100.
        dt, _ = gateway.step(now)
        now += dt
        draft = gateway.p1_decoding[0]
        before = list(draft.seq_state.generated_tokens)
        gateway.on_final_arrive("foreground", 0, [], "Explain distributed caching", 0., now)
        dt, _ = gateway.step(now)
        now += dt
        while gateway.p0_decoding:
            dt, _ = gateway.step(now)
            now += dt
            self.assertEqual(draft.seq_state.generated_tokens, before)
            self.assertEqual(gateway.step_trace[-1].p1_active, 0)
        gateway.step(now)
        self.assertGreater(len(draft.seq_state.generated_tokens), len(before))

    def test_load_gating_and_reuse_budget_use_observed_state(self):
        gateway = PriorityGateway(CalibratedMockEngine(), LoadReuseAdaptiveController())
        conv = ConversationState("c")
        conv.update_partial("Book a table for two", 100.)
        server = gateway.get_server_state(100.)
        adaptive = gateway.controller
        self.assertEqual(adaptive.decide(conv, server).draft_budget_tokens, 12)
        conv.update_accepted_ema(0)
        self.assertLess(adaptive.decide(conv, server).draft_budget_tokens, 12)
        server.p0_active_count = 1
        self.assertFalse(adaptive.decide(conv, server).should_draft)
        self.assertTrue(ReuseOnlyController().decide(conv, server).should_draft)
        server.p0_active_count = 0
        server.p1_queue_len = 1
        self.assertFalse(adaptive.decide(conv, server).should_draft)

    def test_warmup_identities_and_outputs_are_paired_across_all_policies(self):
        traces = load_traces(ROOT / "data/fdb_v2_sample/dev.jsonl")
        schedule = generate_workload_schedule(traces, 4.2, seed=1, target_turns=48)
        harness = ReplayHarness(CalibratedMockEngine(), warmup_turns=6)
        rows = []
        for name in ("no_draft", "fixed_interval", "load_reuse_adaptive", "reuse_only"):
            metrics, summary = harness.run(schedule, make_controller(name), "heavy", 4.2, 1)
            self.assertEqual(summary.num_turns, 42)
            rows.extend(m.to_dict() for m in metrics)
        audit = audit_turn_rows(rows)
        self.assertTrue(audit["artifact_consistent"], audit)
        self.assertEqual(audit["token_comparisons"], 144)

    def test_artifact_audit_detects_tokens_even_when_text_matches(self):
        row = dict(load_regime="light", seed=1, conversation_id="c", turn_id=0,
                   policy_name="no_draft", drafted_tokens=0, accepted_tokens=0,
                   canceled_tokens=0, is_warmup=True, response_text="same", response_tokens=[1])
        changed = {**row, "policy_name": "fixed_interval", "response_tokens": [2], "is_warmup": False}
        audit = audit_turn_rows([row, changed])
        self.assertFalse(audit["artifact_consistent"])
        self.assertEqual(len(audit["output_mismatches"]), 1)
        self.assertEqual(len(audit["warmup_mismatches"]), 1)

    def test_runner_validates_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--mode", "mock",
                   "--loads", "heavy", "--seeds", "1", "--target_turns", "10",
                   "--warmup_turns", "2", "--output_dir", tmp, "--no_timeline"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((Path(tmp) / "manifest_mock_dev.json").read_text())
            self.assertEqual(manifest["status"], "validated")
            validation = json.loads((Path(tmp) / "validation_mock_dev.json").read_text())
            self.assertEqual(validation["reference_checked_turns"], 40)
            self.assertEqual(validation["reference_mismatches"], [])
            retry = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("fresh output_dir", retry.stderr)

    def test_runner_marks_reference_divergence_invalid_and_exits_nonzero(self):
        from scripts import run_experiment

        class BadReferenceEngine(CalibratedMockEngine):
            def generate_greedy_reference(self, history, final_text, max_new_tokens=48):
                return ["different reference"]

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run_experiment.py", "--loads", "light", "--seeds", "1",
                    "--target_turns", "3", "--warmup_turns", "1",
                    "--policies", "no_draft", "--output_dir", tmp, "--no_timeline"]
            with patch.object(sys, "argv", argv), patch.object(run_experiment, "CalibratedMockEngine", BadReferenceEngine), patch("builtins.print"):
                with self.assertRaises(SystemExit):
                    run_experiment.main()
            manifest = json.loads((Path(tmp) / "manifest_mock_dev.json").read_text())
            self.assertEqual(manifest["status"], "invalid_outputs")
            validation = json.loads((Path(tmp) / "validation_mock_dev.json").read_text())
            self.assertEqual(validation["greedy_reference_status"], "failed")
            self.assertEqual(len(validation["reference_mismatches"]), 3)


if __name__ == "__main__":
    unittest.main()
