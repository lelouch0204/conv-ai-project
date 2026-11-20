# Code Review Action Log

Items identified while walking through the infrastructure files one by one.

---

## Pending & Completed Improvements

- **[x] Warmup handling (`scripts/run_experiment.py`, `src/engine/`, `src/simulator/harness.py`)**
  - **Engine warmup (`engine.warmup()`):** Added `warmup()` on `BaseLLMEngine` (no-op on `CalibratedMockEngine`; dummy prefill, batched decode, and verification passes on `VLLMEngine`) and invoked it once after loading the engine in `scripts/run_experiment.py`.
  - **Workload queue & controller warmup (`--warmup_turns`):** Added `--warmup_turns` (default `6`) in `scripts/run_experiment.py` and `warmup_turns` / `is_warmup` in `ReplayHarness` (`src/simulator/harness.py`), marking the first $W$ turns `is_warmup=True` in turn-level JSONL logs and computing summary metrics over steady-state turns.

- **[ ] P1 prefill isolation vs. interleaving (`src/scheduler/gateway.py`)**
  - Currently in `PriorityGateway.step()` Branch 2 (line 378), a `P1` draft prefill can run whenever `p0_waiting_prefill` is empty and batch slots remain, even if `p0_decoding` has active foreground requests (which stalls active `P0` decodes for one prefill duration under load-blind policies like `fixed_interval`).
  - Revisit later whether to keep this realistic contention behavior as default and/or add an option to strictly block `P1` prefills whenever `p0_decoding` is non-empty.

- **[x] Move revision & readiness hooks onto `BaseController` (`src/controllers/base.py`, `src/controllers/policies.py`, `src/scheduler/gateway.py`)**
  - Simplified `ConversationState` to hold raw observations (`previous_partial_text`, `current_partial_text`, `partial_history`, `active_draft_tokens`, `accepted_draft_ema`).
  - Moved `is_revised(conv_state)` and `readiness_score(conv_state)` as default overridable methods onto `BaseController`, and added `discard_existing_draft: bool = False` to `DraftDecision` so `PriorityGateway` obeys controller decisions without inspecting revision cues directly.

- **[x] Standard ML systems GPU serving practices (`src/engine/vllm_engine.py`)**
  - Replaced the custom PyTorch KV-cache batching engine with `VLLMEngine` backed by `vllm` (`enable_prefix_caching=True`, continuous batched `decode_step`, and parallel `prompt_logprobs=1` draft verification).

- **[x] Dataset & streaming trace realism (`src/data/builder.py`, `src/controllers/base.py`)**
  - **Semantic revisions over synthetic ASR jitter:** Removed `compute_prefix_rewrite_distance` (which targeted ASR word-overwrite jitter rather than speaker self-corrections), keeping `detect_revision_cue` focused on mid-utterance semantic pivots on committed/monotonic streaming transcripts.
  - **Independent scenarios per split:** Updated `_expand_scenarios()` in `src/data/builder.py` so each scenario is assigned a single pacing (`fast` or `slow`, balanced per family) rather than duplicating identical conversations across both pacings inside a split.
