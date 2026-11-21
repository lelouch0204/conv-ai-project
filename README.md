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
controller; it has not been tuned. Bounded H100 correctness checks now pass for
the configuration below; this is not a universal equivalence guarantee.

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

## Verified PACE-ICE deployment (November 21, 2026)

The branch is deployed at `~/scratch/conv-ai-project` under the submitting PACE
account. An isolated `.venv` uses the locked dependencies and Python 3.12.5.
Model files are cached under that account's scratch directory. On the existing
PACE checkout, repeat the verified H100 pilot with:

```bash
cd ~/scratch/conv-ai-project
source results/pace_setup/activate.sh
LOADS=light,intermediate,heavy TARGET_TURNS=16 WARMUP_TURNS=4 SEEDS=1 \
  sbatch --partition=coe-gpu --gres=gpu:h100:1 scripts/run_pilot.sbatch
```

This explicitly overrides the script's RTX 6000 resource default. The verified
configuration is H100 80 GB, bfloat16, `VLLM_BATCH_INVARIANT=1`, and model/tokenizer
revision `cfbefacb99257ffa30c83adab238a50856ac3083`. CPU tests pass on PACE (20/20).
Slurm jobs 7225949 (smoke) and 7226125 (three-load pilot) both completed with exit
code 0; all 224 recorded responses match direct greedy references, with no
paired-output, warmup, or accounting errors. Total allocated GPU time was
0.1642 GPU-hours, including initial compilation and warmup.

The pilot evaluates 12 turns per policy/load after four warmup turns, with one
workload seed on the synthetic development sample:

| Policy | Light p50 / p95 (ms) | Intermediate p50 / p95 (ms) | Heavy p50 / p95 (ms) |
| --- | ---: | ---: | ---: |
| `no_draft` | 390.31 / 439.16 | 432.54 / 714.74 | 477.68 / 623.40 |
| `fixed_interval` | 339.71 / 466.19 | 633.91 / 1243.54 | 780.79 / 1119.72 |
| `load_reuse_adaptive` | 372.50 / 467.80 | 352.48 / 651.87 | 493.46 / 753.19 |
| `reuse_only` | 385.82 / 477.22 | 480.95 / 1097.94 | 706.81 / 1310.10 |

The adaptive controller improves on both baselines at intermediate load in this
sample; NoDraft retains the lowest heavy-load p95. These are pilot observations
without estimates of variability, not evidence of a uniform controller win.
Do not compare absolute latency against historical runs with different hardware
or numerical configurations to attribute a speedup to these code fixes.

Artifacts and receipts are in [`results/pace_ice/`](results/pace_ice/): full turn
logs, timelines, reference checks, manifests, HTML replays, CPU test log, pinned
model metadata, and Slurm accounting. The tested code commit is `239c2ea`.

## RTX 6000 collection (November 21, 2026)

These additional jobs explicitly requested **Quadro RTX 6000**, reporting 23,040
MiB and compute capability 7.5. They use the same checkpoint revision and serving
code, with precision recorded separately. Batch invariance is disabled because
[vLLM requires capability 8.0 or newer](https://docs.vllm.ai/en/stable/features/batch_invariance/#hardware-requirements).
The complete collection, including compatibility/diagnostic jobs, used 0.7164
allocated RTX GPU-hours. No H100 was used for these jobs.

### Float16: collected, but output equivalence fails

Job 7226493 ran four policies, three loads, and three workload seeds, with 16
turns per run and four excluded as warmup: 576 responses checked and 432 turns
evaluated for latency. All 36 timed conditions finished, but the strict output
gate correctly exited nonzero and marked the artifacts `invalid_outputs`:

- 73/576 responses disagree with the direct greedy reference: NoDraft 12,
  FixedInterval 20, LoadReuseAdaptive 20, and ReuseOnly 21.
- 35/432 paired comparisons disagree with NoDraft.
- Warmup pairing and draft accounting have zero errors.

These are token-equivalence failures, not measurements of semantic accuracy.
Do not use these latencies as a controlled equivalent-output comparison. The
three-seed means/SDs are retained as diagnostic profiling data in
[`rtx6000_pilot_7226493/aggregate_summary.json`](results/pace_ice/rtx6000_pilot_7226493/aggregate_summary.json).
The earlier eight-turn-per-policy smoke job 7226385 passed; that smaller sample
did not establish correctness for the broader sweep.

Diagnostic job 7226802 reproduces the same-prefix issue: at the sixth response
token, batch size 1 selects `invitation` (ID 43579), while batch size 2 selects
`meeting` (ID 9176). Parallel prompt scoring also selects `meeting`, even though
the single-sequence reference selects `invitation`. This establishes observed
batch/path dependence in this float16 configuration; it does not isolate a
particular kernel defect. The script, scores, and examples are preserved in
[`rtx6000_setup/`](results/pace_ice/rtx6000_setup/).

There were no Triton attention JIT warnings after the first measured-run cache
reset. The float16 timeline nevertheless contains a 5.12-second decode call in
intermediate load, seed 3, FixedInterval. It is retained and contributes to the
large variation; its cause has not been established.

### Float32: sampled output checks pass, with a large latency cost

Float32 light-load check 7226964 passes 64/64 responses. Three-load pilot
7227173 passes 192/192 responses and all 144 paired comparisons. The latter uses
one workload seed and 12 evaluated turns per policy/load after warmup:

| Policy | Light p50 / p95 (s) | Intermediate p50 / p95 (s) | Heavy p50 / p95 (s) |
| --- | ---: | ---: | ---: |
| `no_draft` | 1.882 / 3.817 | 8.339 / 11.639 | 9.090 / 12.315 |
| `fixed_interval` | 3.543 / 5.149 | 8.816 / 13.192 | 11.323 / 14.134 |
| `load_reuse_adaptive` | 1.732 / 3.984 | 8.409 / 11.694 | 9.188 / 12.343 |
| `reuse_only` | 3.522 / 4.979 | 8.932 / 13.135 | 11.297 / 14.108 |

The controller limits naive drafting's latency cost, but does not uniformly beat
NoDraft. Float32 is a separate, slower configuration, not a repair or relabeling
of float16 results. Its load rates were not recalibrated for its service speed;
the high latencies are not evidence of deployable voice performance. Passing
this finite sample does not establish universal numerical invariance, quality,
or held-out performance. Do not pool different precisions or compare these
numbers with H100 to attribute a pure hardware speedup.

Repeat the sampled float32 RTX pilot on the prepared PACE account:

```bash
cd ~/scratch/conv-ai-project
source results/pace_ice/rtx6000_setup/activate_rtx_float32.sh
LOADS=light,intermediate,heavy TARGET_TURNS=16 WARMUP_TURNS=4 SEEDS=1 \
  sbatch --partition=ice-gpu --gres=gpu:rtx_6000:1 scripts/run_pilot.sbatch
```

For the float16 reproduction use `activate_rtx.sh` in the same directory and
`SEEDS=1,2,3`; expect equivalence checks to fail for the recorded configuration.
The complete receipt and accounting are in
[`rtx_collection.json`](results/pace_ice/rtx6000_setup/rtx_collection.json).
Produce the per-seed aggregation table without NumPy or a GPU:

```bash
python scripts/summarize_results.py results/pace_ice/rtx6000_pilot_7226493/summaries_vllm_dev.json
```

This prints the failed validation status alongside the statistics. Means and
sample SDs describe per-seed quantiles, not pooled turn percentiles. The float32
pilot has one seed, so its SD is deliberately null rather than reported as zero.

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
CPU mock-run analysis, PACE deployment, and GPU pilot execution. CPU tests and
artifact/reference checks were executed, including the H100 and RTX trials above.
Team members must review the implementation and GPU results, and record their
own contributions and disclosures for submission.
