# Experiment Plan: Scheduling Early Drafts on a Shared GPU

> **Note:** This document outlines a concrete, literature-backed experimental setup and implementation roadmap based on [CAI_Final_Project.pdf](file:///home/hice1/nparikh44/scratch/conv-ai-project/CAI_Final_Project.pdf). Specific implementation choices (e.g., serving backend, model checkpoint, thresholds) can be adjusted based on pilot measurements.

---

## 1. Research Motivation & Core Tradeoff

### Why Early Drafting Still Matters on a Shared GPU
At first glance, early drafting during user speech can seem counterproductive when multiple conversations share a GPU. However, two structural properties of LLM voice serving make it worthwhile under the right conditions:

1. **TTS & Sequential Generation Bottleneck:** A voice assistant cannot start speaking audio on the very first token. Text-to-Speech (TTS) typically needs a full clause or sentence (~15–30 tokens), and agents that emit brief reasoning or intent/tool tokens need even more before speaking. Autoregressive decoding of $N$ tokens after `utterance_end` requires $N$ sequential forward passes—extra GPU FLOPs cannot parallelize token $t+1$ with token $t$. Early drafting shifts those sequential steps into the user's speaking window, reducing post-speech latency to **one parallel verification forward pass** when the draft is valid.
2. **Batch Piggybacking in Memory-Bandwidth-Bound Regimes:** At low-to-moderate batch sizes, LLM decode steps are dominated by loading model weights from HBM rather than arithmetic compute. Adding background draft steps alongside active foreground decoding can reuse the weight load with modest step-time overhead—up to the compute/KV-cache saturation knee.

### Why Unthrottled Early Drafting Fails on a Shared GPU
Unlike single-user settings where listening-time compute is free:
1. **KV-Cache Bandwidth Scales with Batch Size:** Each conversation holds an independent multi-turn KV cache. Extra draft sequences increase HBM KV-cache reads during every decode step, slowing down foreground users who have already finished speaking.
2. **Changing Transcripts Trigger Compute-Bound Prefills:** Unlike standard speculative decoding (fixed prompt, appended draft tokens), streaming speech updates change the user prompt prefix. Re-prefilling updated prefixes and verifying drafts are compute-bound operations that can stall concurrent decodes.
3. **Mis-speculation Externalities:** When User A revises or pivots mid-utterance, the wasted draft compute directly consumes batch slots and adds queueing delay to Users B and C.

### Research Questions (from Proposal)
- **RQ1:** How do server load and later changes to a spoken request affect the benefit of early drafting?
- **RQ2:** Do transcript edits or semantic-readiness signals improve scheduling beyond server load and recent draft reuse?
- **RQ3:** Do any latency gains persist across complete conversations without reducing task completion or correction handling?

---

## 2. Literature Synthesis & Experimental Inspiration

| Paper | Core Mechanism / Idea | Experimental Setup Patterns Adopted |
| :--- | :--- | :--- |
| **PredGen** (Li & Grover, 2025) | Input-time speculation during speech with interleaved verification and KV-cache reuse | - Timestamped incremental prefix streams (`simulated_inputs.json`) representing streaming ASR chunks.<br>- Greedy verification against final prompt + KV-cache slicing to longest matching prefix.<br>- **Time-To-First-Sentence (TTFS)** (delay from `utterance_end` to first complete validated sentence) as primary latency metric. |
| **LTS-VoiceAgent** (Zou et al., 2026) | Listen-Think-Speak orchestration with dynamic semantic triggers | - **Pause-and-Repair** stress testing: mid-utterance pauses, disfluencies, and self-corrections that tempt premature drafting.<br>- Tracking rollback frequency, token waste, and accuracy vs. post-turn silent gap. |
| **Full-Duplex-Bench-v2** (Lin et al., 2025) | Multi-turn streaming dialogue benchmark with an Automated Examiner | - **Daily**, **Correction**, and **Entity Tracking** task families under **Fast** and **Slow** pacing.<br>- External Automated Examiner driving full multi-turn conversations where turn $t+1$ responds to the agent's actual output from turn $t$. |
| **TurboSpec** (Liu et al., 2025) | Closed-loop speculation control for LLM serving goodput | - Open-loop **Poisson arrival sweep** ($\lambda_{\text{light}}, \lambda_{\text{mid}}, \lambda_{\text{heavy}}$) across memory-bound, knee, and compute/queue-saturated regimes.<br>- Measuring **Goodput** (accepted/valid tokens per second) alongside tail latency. |
| **AdaServe** (Li et al., 2026) | Multi-SLO serving with customized speculation | - **SLO attainment** (% of turns meeting a target TTFS deadline) and throughput/turns-per-second meeting the latency target. |

---

## 3. System Architecture

The experiment uses a **two-stage evaluation harness** sharing a single serving engine and controller interface:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Trace Replay Harness (RQ1 & RQ2 — Scheduling & Contention)      │
│                                                                          │
│  Poisson Multi-Client Replayer (λ = Light, Mid, Near-Saturation)         │
│   ├── Client 1: (t_1, P_1) -> (t_2, P_2) -> ... -> (t_end, P_final)      │
│   ├── Client 2: (t_1, P_1) -> [revision/pivot] -> (t_end, P_final)       │
│   └── Client M: ...                                                      │
│                         │                                                │
│                         ▼                                                │
│  Shared-GPU Gateway & Controller                                         │
│   ├── Priority 0 Queue (Foreground): Completed turns (Verify + Decode)   │
│   ├── Priority 1 Queue (Background): Optional drafts (<= 1/conversation) │
│   └── Pluggable Policy: decide(conv_state, server_state) -> (bool, K)    │
│                         │                                                │
│                         ▼                                                │
│  Shared LLM Engine (1 Frozen Instruction-Tuned Model on 1 GPU)           │
│   └── Greedy Prefix Verifier (output strictly == non-speculative greedy) │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Interactive Multi-Turn Harness (RQ3 — Conversation Quality)     │
│                                                                          │
│  External Automated Examiner (outside measured GPU) <-> Benchmark Adapter│
│   └── Drives Daily / Correction / Entity Tracking multi-turn dialogues   │
│       against the exact same Gateway + Engine + Controller policies.     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Serving & Verification Design Rules
1. **Foreground Priority & Draft Caps:** Completed turns (`Priority 0`) always take precedence over optional drafts (`Priority 1`). At most one active draft exists per conversation; obsolete transcript versions are immediately canceled.
2. **Exact Greedy Verification & Cache Hygiene:**
   - When `FINAL(conv_id, P_final)` arrives and a draft $R = (r_1, \dots, r_m)$ exists from an earlier partial prompt:
     - Run **one parallel verification forward pass** on `[P_final + R]`.
     - Find the longest prefix $r_1 \dots r_k$ matching greedy `argmax` logits under `P_final`.
     - Retain KV cache for `[P_final + r_1 ... r_k]` and resume autoregressive decoding from position $k+1$ (or return immediately if the first validated sentence is already complete).
   - Never treat KV-cache entries generated under a different prompt as valid final-prompt state without verification.
   - Never expose unverified draft tokens to the user/client.
3. **Engine Implementation Note:** A lightweight custom batched PyTorch/HuggingFace loop (using `DynamicCache` and FlashAttention-2 / SDPA) makes priority preemption, draft-prefix verification over changing user prompts, and exact wasted-token accounting straightforward. (Other serving backends such as vLLM with custom verification/scoring remain viable alternatives if needed.)
4. **Model Size Check in Pilot:** The proposal targets a frozen 1B–3B instruction-tuned LLM on one A100/H100 GPU. During the pilot, verify whether a 1B–3B model produces a large enough non-speculative TTFS (time to generate the first complete sentence) and adequate FDB-v2 task accuracy—or whether concurrency/context length (or a larger checkpoint / brief structured pre-response step) is needed so the scheduling tradeoff is clear and realistic.

---

## 4. Experimental Design

### A. Streaming Trace Format (Stage 1)
Convert **Full-Duplex-Bench-v2** scenarios (**Daily**, **Correction**, **Entity Tracking**) into timestamped partial-transcript JSONL traces, split cleanly between **development** and **held-out** sets (no utterance prefixes crossing the split):

```json
{
  "conversation_id": "conv_01",
  "turn_id": 0,
  "family": "correction",
  "synthetic_revision": true,
  "partials": [
    {"t_ms": 400, "text": "Book a table for two"},
    {"t_ms": 900, "text": "Book a table for two at 7pm"},
    {"t_ms": 1400, "text": "Book a table for two at 7pm actually make it 8pm"}
  ],
  "final": {"t_ms": 1600, "text": "Book a table for two at 7pm, actually make it 8pm."}
}
```
- The controller never sees future transcripts or `final.t_ms` in advance.
- Recorded speech will be transcribed ahead of time where available; constructed revision traces will be explicitly labeled `synthetic_revision: true`.
- Separate analyses by **steady speech**, **pause-and-repair / mid-utterance corrections**, **pacing** (fast vs. slow), and **history length**.

### B. Controller Policies (Comparison Matrix)
All policies implement `decide(conv_state, server_state) -> DraftAction(should_draft: bool, draft_budget: int)` and run on the exact same model, prompts, hardware, verification, and output limits:

| # | Policy | Description | Role in Evaluation |
| :--- | :--- | :--- | :--- |
| **1** | `NoDraft` | Never drafts during speech; runs only Priority 0 after `utterance_end`. | Baseline (non-speculative greedy reference) |
| **2** | `FixedInterval` | Drafts fixed budget $K$ every $\Delta T$ ms during incoming speech. | Baseline (naive PredGen-style schedule) |
| **3** | `SemanticOnly` | Triggers fixed budget $K$ when a lightweight readiness heuristic/classifier indicates an answerable prefix. | Baseline (LTS-VoiceAgent-style semantic trigger) |
| **4** | `LoadReuseAdaptive` | Foreground priority + drafts only when Priority 0 queue backlog is below threshold, scaling $K$ via recent EMA of accepted draft tokens. | Core resource-aware controller |
| **5** | `LoadReuse + Edit` | Adds transcript-edit history / instability suppression (e.g., recent word rewrites or pivot markers) to (4). | Tests RQ2 (edit signal value beyond load/reuse) |
| **6** | `LoadReuse + Semantic` | Adds the semantic readiness trigger from (3) to (4). | Tests RQ2 (semantic signal value beyond load/reuse) |
| **7** | `ReuseOnly` *(Ablation)* | Removes load adaptation from (4) while keeping foreground priority / reuse scaling. | Ablation isolating load adaptation |

*Note:* Controller and semantic-trigger compute/latency overheads are included in end-to-end measurements.

### C. Traffic Calibration & Schedule
- Calibrate three Poisson arrival rates ($\lambda_{\text{light}}, \lambda_{\text{mid}}, \lambda_{\text{heavy}}$) empirically on the target GPU (A100/H100 on PACE):
  - **Light:** Low active batch size; GPU under-utilized during listening.
  - **Intermediate:** Batch sizes near the compute/KV-bandwidth knee where unthrottled drafts start impacting neighbors.
  - **Near-Saturation:** High arrival rate with Priority 0 queueing; naive drafting causes tail-latency spikes and timeouts.
- Run final held-out comparisons across at least **3 independent workload schedules** with paired scenarios across policies.

### D. Metrics
- **Primary Latency & Goodput Metrics:**
  - Median (p50) and 95th-percentile (p95) delay from `utterance_end` to **first validated answer sentence** (TTFS).
  - Turns per second meeting a stated response latency target (SLO attainment).
- **Efficiency & System Overhead Metrics:**
  - Accepted draft tokens, draft acceptance ratio ($\text{accepted} / \text{drafted}$), canceled work, timeouts, Priority 0 queue wait time, peak GPU memory, and allocated GPU-hours.
- **Conversation Quality Metrics (Stage 2 / Interactive):**
  - Dialogue-level task scores, correction handling, and entity-tracking accuracy (with conversation-level confidence intervals and blinded manual audit of a sample of automated scores).

---

## 5. Implementation Roadmap (PACE Cluster)

Compute available: PACE cluster with Slurm (`sbatch` / `srun`), targeting 1 NVIDIA A100/H100 GPU (5–10 GPU-hours for pilot calibration, then scaled based on measured runtimes).

### Phase 1: Repository Scaffold & Exact Greedy Verification Engine
- Set up repo layout (`configs/`, `data/`, `src/engine/`, `src/scheduler/`, `src/controllers/`, `src/simulator/`, `scripts/`).
- Implement batched LLM wrapper with `DynamicCache` slicing and parallel draft verification.
- Add unit tests verifying that verified greedy outputs match non-speculative greedy decoding token-for-token across prompt-change scenarios (no prefix match, partial match, full match).

### Phase 2: Trace Generator & Event-Driven Replay Simulator
- Ingest/adapt Full-Duplex-Bench-v2 scenarios (Daily, Correction, Entity Tracking) into dev and held-out partial-transcript JSONL traces.
- Build the multi-client Poisson trace replayer, Priority 0 / Priority 1 gateway, cancellation logic, and telemetry logger (TTFS, queue wait, wasted tokens).

### Phase 3: Controllers & Three-Load Pilot on PACE
- Implement the baseline, adaptive, edit-history, semantic-trigger, and ablation controllers.
- Write Slurm batch scripts to run a 3-load calibration pilot on 1 A100/H100 GPU, tune dev budgets/thresholds, and inspect whether the 1B–3B model exhibits clear contention + measurable TTFS gains.

### Phase 4: Interactive Dialogue Adapter & Held-Out Evaluation
- Build the benchmark adapter connecting the external Automated Examiner (running outside the measured GPU workload) for complete multi-turn conversations (RQ3).
- Run held-out evaluation across $\ge 3$ workload seeds, compute run-level latency variation and conversation-level confidence intervals, perform blinded manual audits, and run failure analysis (corrections, pacing, history length).
