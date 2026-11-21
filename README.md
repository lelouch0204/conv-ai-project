# Scheduling early drafts on a shared GPU

This research prototype replays partial transcripts into a shared language-model
scheduler. It tests when drafting before the end of a spoken request reduces
post-utterance latency and when wasted drafts delay other requests. There is no
live ASR, TTS, or interactive examiner yet.

## Current implementation

- `no_draft`: wait for the final transcript.
- `fixed_interval`: load/speech-blind drafting on partial arrivals, with a 500 ms
  cooldown, at least three words, and a 20-token budget. It is not a periodic
  timer during silence. The earlier implementation also discarded candidates
  on lexical revision cues; that confound has been removed.
- `load_reuse_adaptive`: draft only with no queued or active foreground work;
  cap queued/active background requests globally at one. Pause background
  execution while foreground is active. Size drafts using the conversation's
  EMA of accepted lengths (initial 12, alpha 0.4, bounded to 4–20 tokens).
- `reuse_only`: the same reuse budget and cooldown, with load gates removed.

All policies use the same gateway and verification path. Kernel calls already
in flight cannot be preempted. The adaptive policy is a conservative first
controller; it has not been tuned or validated on a GPU.

## Setup and checks

The cluster environment uses Python 3.12 and `uv.lock`. On an allocated NVIDIA
GPU, install the locked environment with `uv sync --frozen --extra dev`.

For CPU-only tests, a Python environment with NumPy is sufficient; imports of
vLLM are lazy. Run:

```bash
python -m unittest discover -s tests -v
python scripts/audit_results.py results/rtx_6000/turns_vllm_dev.jsonl
```

The stdlib-only audit compares paired outputs and warmup identities and checks
token accounting. `--strict` exits nonzero on inconsistencies. Old results only
contain text, so this audit cannot establish token equivalence or independently
check a greedy reference.

## Reproduce the new CPU experiment

```bash
python scripts/run_experiment.py --mode mock \
  --policies no_draft,fixed_interval,load_reuse_adaptive,reuse_only \
  --loads light,intermediate,heavy --seeds 1,2,3 \
  --target_turns 48 --warmup_turns 6 --no_timeline \
  --output_dir results/my_mock_run
```

Use a fresh output directory for every invocation. The runner saves turn logs,
summaries, a source/configuration manifest, and output-validation diagnostics.
Warmup is determined by the first six **final arrivals** in the offered workload,
so every policy evaluates the same 42 of 48 turns per run. Warmup-only runs are
not reported as evaluated data. Prefix caches reset before every policy run.

After all timed runs, outputs are compared token-for-token against direct greedy
generation and against `no_draft` when present. Checks include warmup turns.
Mismatch leaves artifacts on disk, marks the manifest `invalid_outputs`, and
exits nonzero. `--skip_output_validation` is for diagnostics and marks successful
runs `unvalidated`. A manifest left `running` or `initializing` is incomplete.
Reference checks are outside the measured replay and cannot warm later runs.
Passing finite checks is evidence for the sampled prompts/configuration, not a
universal proof of floating-point invariance.

## GPU rerun

Submit from the repository on PACE, where the prepared model cache is available:

```bash
sbatch scripts/run_pilot.sbatch
```

The default is a 16-turn, one-seed smoke test of four policies on RTX 6000,
float16. It creates a job-specific output directory. Override the partition/GPU
request to match the cluster allocation. After correctness checks pass, expand:

```bash
TARGET_TURNS=48 WARMUP_TURNS=6 SEEDS=1,2,3 sbatch scripts/run_pilot.sbatch
```

The script uses the submitting account's `~/scratch/.cache` for caches and
preserves explicitly supplied `HF_HOME`, `UV_CACHE_DIR`, and `XDG_CACHE_HOME`.
Override `PACE_CACHE_ROOT` for another scratch location or `PROJECT_DIR` when
submitting outside the checkout. Its default `HF_HUB_OFFLINE=1` requires the model
to be downloaded into that account's cache before the job starts; set
`HF_HUB_OFFLINE=0` explicitly only when downloads are intended. Load the prepared
Python environment and ensure `uv` is on `PATH` before submitting.
Set `MODEL_REVISION` to pin both model and tokenizer for batch jobs; `MODEL_NAME`
can select another checkpoint. Both values are passed to the experiment runner.

Model and tokenizer revisions can be pinned with the runner's
`--model_revision`; if omitted, the model name alone does not identify immutable
weights. `--max_new_tokens` defaults to 48. On supported configurations, consult
[vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)
when investigating numerical differences. Do not assume temperature zero makes
different batch shapes or scoring/generation paths identical.

Generate the interactive report for any completed run:

```bash
python scripts/visualize_run.py results/my_mock_run/turns_mock_dev.jsonl --ascii
```

## Interpretation and remaining limitations

The original GPU results are unchanged. Their summaries agree with their logs,
but the audit finds 4/48 paired text mismatches and two warmup-membership
differences in RTX runs, and 1/8 text mismatches in the earlier H100 smoke test.
These artifacts are provisional, not validated equivalent-output comparisons.
The fixes reject missing rank-1 verification scores, cap accepted output length,
and make unpenalized greedy sampling explicit. The cause of the recorded GPU
divergence still needs hardware diagnosis; CPU stubs do not establish its fix.

The vLLM backend creates a new completed one-token request on every step, with
prefix caching. It is not persistent continuous decoding. A production serving
comparison requires addressing that overhead. Replay time sums engine-call
durations; controller, tokenization, and logging outside those calls are excluded.
The existing `ttfs` metric is a first-answer-segment proxy: punctuation after at
least four tokens (with selected abbreviations excluded), a 24-token fallback,
or response completion. It is not measured audio latency. Timeline busy time is
scheduled call occupancy, not measured GPU utilization; splitting mixed batches
between P0/P1 by sequence count is an attribution heuristic.

Load labels use fixed offered rates (0.25, 1.6, 4.2 turns/s), implemented as
Poisson conversation starts with fixed within-conversation gaps. They are not
calibrated capacity points. Finite-run goodput uses the interval from the first
evaluated final arrival to the last evaluated completion, not a steady-state
capacity estimate. The mock uses synthetic timing and response templates and
ignores dialogue history; mock improvements are not hardware or quality results.

Remaining research work includes persistent serving, exact-output GPU checks,
hardware load calibration, semantic/edit ablations, actual benchmark ingestion,
held-out evaluation with variability, and interactive multi-turn quality checks.

## Data and compute statement

`data/raw_scenarios.json` contains repository-authored synthetic dialogues in
Daily, Correction, and Entity Tracking styles. The checked-in sample has 16 dev
turns and 15 held-out turns across nine conversations each; it is not an imported
Full-Duplex-Bench-v2 evaluation. Timestamps and correction pauses are synthetic.
No live participant data or external dataset license is supplied here. Upstream
data provenance/licensing must be documented when a real importer is added.

Historical GPU artifacts are under `results/` and `results/rtx_6000/`. Their
GPU-hours were not recorded in those JSON files. New manifests record available
hardware/software/configuration details; preserve Slurm accounting to report
allocated GPU-hours. `results/review/mock_dev/` contains CPU simulation only.

## AI assistance

OpenAI Codex assisted with code review, implementation, regression tests, and
CPU mock-run analysis. CPU tests and artifact checks were executed; changes have
not been validated on a GPU in this workspace. Team members must review the
implementation, verify GPU results, and record their own contributions and
disclosures for submission.
