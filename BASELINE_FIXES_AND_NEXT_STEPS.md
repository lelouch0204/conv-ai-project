# Baseline correctness fixes and load/reuse scheduling

Date: November 21, 2026

Branch: `fix/baseline-correctness-and-load-reuse`

## Purpose and current status

This document explains the issues found in the original baseline experiments,
why they matter, the changes implemented locally, and the evidence still needed
before using the new implementation for GPU performance claims.

The original `no_draft` policy correctly avoids speculative work. However, the
saved GPU comparisons do not establish the project's intended guarantee that
every policy produces the same final-prompt greedy answer. Correct scheduling
logic alone does not establish a correct or fair experiment.

The fixes and two additional controllers are committed and deployed on PACE-ICE.
All 20 CPU tests pass locally and on PACE. The mock sweep and bounded H100 GPU
runs pass output and accounting checks. Section 10 records the exact GPU
validation scope; historical GPU artifacts have been preserved unchanged.

## 1. Evidence from the original experiments

The saved summaries agree with their underlying turn logs. The problems concern
output equivalence, experimental pairing, and interpretation of measurements.

| Artifact set | Paired turns | Response-text mismatches | Paired turns with different warmup flags |
| --- | ---: | ---: | ---: |
| Earlier H100 smoke test | 8 | 1 | 0 |
| RTX 6000 pilot | 48 | 4 | 2 |

Reproducible audit reports are available in
[`results/review/audit_h100.json`](results/review/audit_h100.json) and
[`results/review/audit_rtx6000.json`](results/review/audit_rtx6000.json).
These historical logs contain response text rather than token IDs. A text
mismatch proves disagreement, but matching text does not prove token equality.

Consequently, the original GPU measurements can describe prototype behavior,
but should not be presented as validated equivalent-output latency comparisons.
The precise cause of the recorded GPU divergence remains unresolved.

## 2. Verification must reject missing evidence

**Problem.** When prompt log probabilities were missing at a candidate-token
position, the verifier could fall back to the token generated after the entire
draft. That token is conditioned on the draft's potentially invalid suffix,
rather than only the final prompt and accepted prefix.

The old helper also selected the best token from an entry without establishing
that the actual rank-1 token was included. An entry containing only the
candidate's own score is insufficient evidence of greedy agreement.

**Implemented change.** Verification now requires a rank-1 score at each
candidate position it examines. If that evidence is unavailable, it raises an
error instead of releasing a potentially invalid token. At a known mismatch,
the bonus token comes from the greedy prediction at that mismatch position.

**Why this is necessary.** Speculative scheduling is only comparable with
`no_draft` if accepted output is valid under the final request. A verification
failure must not silently become a different generation algorithm.

Implementation: [`src/engine/vllm_engine.py`](src/engine/vllm_engine.py).

## 3. Sampling and output limits must be consistent

**Problem.** Implicit generation settings obscure whether prompt scoring and
generation use the same decision rule. The verifier could also accept more
candidate tokens than the requested response limit.

**Implemented changes.**

- Use explicit unpenalized greedy sampling throughout: temperature 0,
  repetition penalty 1, presence/frequency penalties 0, top-p 1, top-k -1,
  and min-p 0.
- Disable automatic model-specific generation defaults and supply the
  collected stop-token IDs explicitly.
- Limit verification to the requested output length, preserve accounting for
  the full supplied draft, and avoid adding a bonus token past the limit.
- Expose `--model_revision` and apply it to both model and tokenizer loading.

**Why this is necessary.** Different sampling rules or response limits confound
latency comparisons. Revision pinning also makes an experiment reproducible;
the user must supply a revision to obtain an immutable checkpoint identifier.

These changes eliminate logical/configuration ambiguities. They do not prove
numerical equivalence between all GPU batch shapes or between prompt scoring
and autoregressive generation. vLLM documents that reproducibility is not
guaranteed by default; see its
[reproducibility guidance](https://docs.vllm.ai/en/stable/usage/reproducibility/).

## 4. Warmup exclusion must preserve paired evaluation

**Problem.** The first W completed turns were excluded as warmup. Completion
order depends on the policy, so the policies could be evaluated on different
turns. This occurred in the original heavy-load RTX comparison.

**Implemented change.** Warmup identities are selected from the first W final
arrivals in the offered workload, with deterministic tie breaking. Every policy
excludes those same conversation/turn identities. A workload containing only
warmup turns reports zero evaluated turns instead of silently including them.

**Why this is necessary.** A policy should not improve its reported latency by
changing which examples enter the evaluated sample. The warmup rule is an
evaluation decision; future final-arrival information is not exposed to the
controller.

Implementation: [`src/simulator/harness.py`](src/simulator/harness.py).

## 5. Cache state and experiment provenance must be controlled

**Problem.** A single engine was reused across policies without resetting its
prefix cache. Earlier runs could warm prompts for later runs. The original
artifact schema also omitted response token IDs and important run metadata.

**Implemented changes.**

- Reset prefix caches before each policy/load/seed run. The GPU backend refuses
  to proceed if it cannot perform the reset.
- Save response token IDs in addition to text.
- Save arguments, Git commit/status, source hashes, dataset hash, available
  hardware/software information, cache treatment, and metric definitions.
- Require a fresh output directory to prevent accidental artifact replacement.
- Save completed runs incrementally so later failures retain diagnostic data.

**Why this is necessary.** Paired workloads need comparable starting conditions,
and a result must identify the code, input, and configuration that produced it.
Allocated GPU-hours still need to be recovered from Slurm accounting.

Implementation: [`scripts/run_experiment.py`](scripts/run_experiment.py) and the
engine cache-reset interface.

## 6. Output validation must be an experiment gate

**Problem.** Previous experiments could save and summarize divergent answers
without failing. Existing mock/stub tests did not establish real-GPU equality.

**Implemented change.** After all timed work, the runner compares every recorded
response token sequence with a direct greedy reference for its final prompt.
It also checks agreement with `no_draft` where present, warmup pairing, duplicate
or missing paired records, and draft-token accounting. Reference generation runs
after measurements so it cannot warm later measured runs.

If output checks fail, the runner retains diagnostic artifacts, records
`invalid_outputs` in the manifest, and exits nonzero. Successful runs that use
`--skip_output_validation` are explicitly marked `unvalidated`. An interrupted
run left `initializing` or `running` is incomplete.

**Why this is necessary.** A faster response generated by a different algorithm
does not establish a scheduling improvement under the stated equivalence rule.
Passing these checks is evidence for the sampled configuration and prompts,
not a universal proof of floating-point invariance.

Implementation: [`src/validation.py`](src/validation.py),
[`scripts/audit_results.py`](scripts/audit_results.py), and the experiment runner.

## 7. Keep the fixed-interval baseline independent of speech signals

**Problem.** `fixed_interval` could discard a candidate after detecting a lexical
revision cue. That incorporates a speech-derived signal into the baseline that
is meant to isolate load- and speech-blind drafting.

**Implemented change.** Remove that revision-specific behavior. The baseline
retains its three-word minimum, 500 ms cooldown, and 20-token budget. Draft
requests are triggered by partial arrivals; this is not a periodic timer that
keeps firing during silence. The shared verifier checks any retained candidate
against the final prompt.

**Why this is necessary.** Subsequent edit/readiness experiments need a clearly
defined baseline so their signal contribution can be isolated.

This changes baseline behavior. New results must be labeled as coming from the
revised implementation rather than replacing or relabeling historical runs.

## 8. Add a conservative load/reuse controller and an ablation

Two policies have been added in
[`src/controllers/policies.py`](src/controllers/policies.py).

### `load_reuse_adaptive`

- Admit drafts only when there are no queued or active foreground requests.
- Cap queued/active background requests globally at one.
- Pause background prefill/decode work at scheduling boundaries while
  foreground work is active; resume eligible work when foreground drains.
- Choose a draft budget from the conversation's EMA of accepted draft lengths:
  initial estimate 12 tokens, update alpha 0.4, bounded to 4–20 tokens.
- Use only observed state and previous reuse, without future transcripts or
  semantic/revision signals.

The gateway now consults a controller execution hook as well as admission
decisions. Admission checks alone would let an already-started draft continue
consuming GPU work after foreground demand appears. An in-flight engine call
cannot be preempted; cancellation/capacity eviction can still remove a draft.

### `reuse_only`

Use the same cooldown and EMA-derived draft budget with load gating removed.
This is the ablation needed to test whether load control adds value beyond
simply reducing draft budgets.

**Why these additions are useful.** They implement a first test of the project's
core scheduling question without introducing a learned predictor or changing
the language model. The conservative idle-only rule is a starting point, not an
optimized controller or evidence that load gating always helps.

## 9. Validation and new mock results

All 20 CPU tests pass, including regressions covering missing scores, mismatch
positions, changed prompts, output caps, EOS, cache resets, background pausing,
paired warmup identities, and the runner's failure status.

The new sweep contains four policies × three loads × three seeds × 48 turns:
1,728 replayed turns. Six warmup turns per run leave 1,512 evaluated turns.
All 1,728 outputs, including warmup, match the mock greedy reference. There are
zero paired-output, warmup-membership, or accounting mismatches.

Heavy-load mock results, with means of per-seed quantiles:

| Policy | Mean median TTFS (ms) | Mean p95 ± sample SD across three seeds (ms) |
| --- | ---: | ---: |
| `no_draft` | 158.3 | 197.2 ± 8.9 |
| `fixed_interval` | 30.1 | 260.7 ± 100.0 |
| `load_reuse_adaptive` | 64.4 | 189.4 ± 13.0 |
| `reuse_only` | 72.8 | 165.1 ± 19.5 |

The load/reuse controller reduces fixed drafting's heavy-load tail, while fixed
drafting has lower median latency. Reuse-only has the best heavy-load tail in
this mock sweep. The evidence therefore does not establish that the load gate
outperforms its ablation. No parameters were tuned to reverse that outcome.

Artifacts: [`results/review/mock_dev/`](results/review/mock_dev/), including the
manifest, per-run summaries, turn logs, validation report, and HTML replay.
These are synthetic simulation results, not GPU measurements. The mock ignores
dialogue history and uses deterministic response templates and synthetic timing.

## 10. PACE-ICE deployment and GPU validation

Code commit `239c2ea` was deployed into the submitting user's
`~/scratch/conv-ai-project` checkout. Python 3.12.5, an isolated `uv` tool
environment, and the locked project `.venv` were installed without changing
other project environments. The Slurm script now uses account-local cache paths
and accepts `MODEL_NAME` and `MODEL_REVISION`; it no longer assumes another team
member's home directory.

The tested configuration uses one NVIDIA H100 80 GB GPU, bfloat16, vLLM 0.34.0,
and `VLLM_BATCH_INVARIANT=1`. Model and tokenizer are pinned to
`cfbefacb99257ffa30c83adab238a50856ac3083` of
`microsoft/Phi-4-mini-instruct`.

| Slurm job | Scope | Checked responses | Result |
| --- | --- | ---: | --- |
| 7225949 | Four policies, light load, eight turns each | 32 | Completed, reference checks passed |
| 7226125 | Four policies, three loads, 16 turns each | 192 | Completed, reference checks passed |

Both jobs exited 0. All 224 responses, including warmup, match their direct
greedy references. The artifact checks find zero output, warmup-pairing, or
accounting mismatches. Slurm records 0.1642 allocated GPU-hours across both jobs,
including first-use compilation and warmup.

The three-load pilot has one seed and 12 evaluated turns per policy/load after
four warmup turns. At intermediate load, `load_reuse_adaptive` has median/p95
352.48/651.87 ms, compared with NoDraft's 432.54/714.74 ms and fixed drafting's
633.91/1243.54 ms. At heavy load, its p95 is 753.19 ms versus fixed drafting's
1119.72 ms, but NoDraft is better at 623.40 ms. Thus the controller helps in some
sampled conditions, but does not uniformly beat waiting until the end.

These results establish sampled output agreement for this H100 configuration.
They do not diagnose the historical divergence or prove that it is resolved on
RTX 6000 or other numerical configurations. They also do not establish
statistical significance, held-out generalization, or conversation quality.

Copied artifacts are in [`results/pace_ice/`](results/pace_ice/). See
[`setup/deployment.json`](results/pace_ice/setup/deployment.json) for the
deployment receipt and [`README.md`](README.md) for the full result table and
repeat-run instructions. The remote checkout additionally contains
`results/pace_setup/activate.sh` for restoring the tested environment.

## 11. What remains before making stronger claims

1. Expand the bounded H100 pilot across workload seeds and held-out inputs, and
   investigate mismatches if other GPU/configurations are used. Do not bypass
   failing checks for reported results.
2. Replace or characterize the current backend's repeated one-token requests.
   It uses prefix caching, not persistent vLLM decode requests. This is a
   material serving-overhead limitation.
3. Calibrate hardware load regimes and SLOs. Existing rates are fixed offered
   rates, not established underutilized/knee/saturated operating points.
4. Include or separately measure controller and other host-side overheads.
   The replay clock currently sums timed engine-call durations only.
5. Refine the first-sentence metric. Current TTFS is a first-answer-segment proxy
   with punctuation, a 24-token fallback, or response completion; it excludes
   ASR/TTS. Dashboard occupancy is a scheduling proxy, not hardware utilization.
6. Add semantic/edit ablations, actual benchmark ingestion, frozen held-out
   comparisons with variability, and an interactive multi-turn quality audit.

## Reproduction commands

CPU tests, from an environment with NumPy:

```bash
python -m unittest discover -s tests -v
```

Audit original GPU artifacts without requiring NumPy or a GPU:

```bash
python scripts/audit_results.py results/rtx_6000/turns_vllm_dev.jsonl --strict
```

This command is expected to exit nonzero because the historical artifacts fail
the paired-output and warmup checks.

Repeat the new mock sweep into a fresh directory:

```bash
python scripts/run_experiment.py --mode mock \
  --policies no_draft,fixed_interval,load_reuse_adaptive,reuse_only \
  --loads light,intermediate,heavy --seeds 1,2,3 \
  --target_turns 48 --warmup_turns 6 --no_timeline \
  --output_dir results/my_review_run
```

GPU smoke test, submitted from the repository in the prepared PACE environment:

```bash
sbatch scripts/run_pilot.sbatch
```

After GPU correctness checks pass, expand the sweep:

```bash
TARGET_TURNS=48 WARMUP_TURNS=6 SEEDS=1,2,3 sbatch scripts/run_pilot.sbatch
```

See [`README.md`](README.md) for setup, data/compute limitations, and the
AI-assistance disclosure. Human reviewers must verify the GPU findings before
using them in the milestone or final paper.
