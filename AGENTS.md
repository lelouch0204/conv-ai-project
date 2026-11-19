# Project Context Notes (`AGENTS.md`)

> **IMPORTANT — Non-Authoritative Reference:**
> This file is provided strictly as **informational background** to help orient collaborators and AI coding assistants. **Nothing in this file is prescriptive or binding.** Teammates own different parts of the project and may choose completely different architectures, serving frameworks, controller formulations, or experimental designs. Do not treat suggestions or tentative plans referenced here as rigid constraints—defer to the active user's instructions and the team's evolving decisions.

---

## Project Overview

- **Course:** ECE 6256 / CS 6604 (Fall 2026)
- **Proposal Document:** [CAI_Final_Project.pdf](file:///home/hice1/nparikh44/scratch/conv-ai-project/CAI_Final_Project.pdf)
- **Topic:** *"When Should a Voice Agent Think Ahead? Scheduling Early Drafts on a Shared GPU"*
- **Core Problem:** Voice assistants can lower response latency by speculatively drafting answers during unfinished user speech, but later words can invalidate earlier drafts and speculative work shares GPU capacity with completed-turn responses across concurrent conversations.

### Research Questions (from Proposal)
1. **RQ1:** How do server load and later changes to a spoken request affect the benefit of early drafting?
2. **RQ2:** Do transcript edits or semantic-readiness signals improve scheduling beyond server load and recent draft reuse?
3. **RQ3:** Do any latency gains persist across complete conversations without reducing task completion or correction handling?

---

## Team Areas (from Proposal)

- **Abhishek Revinipati:** Serving and profiling
- **Vedant Mohanty:** Data and dialogue adapter
- **Nandan Bharatkumar Parikh:** Controller and baselines
- **Venkatesan Sundar:** Experiments, quality audit, and reproducibility

*(All members contribute to literature verification, analysis, and writing.)*

---

## Tentative Principles & Evaluation Scope (from Proposal)

- **Single Shared GPU:** Serving a frozen instruction-tuned LLM (targeted around ~1–3B parameters, subject to pilot findings) across multiple concurrent conversations.
- **Safety / Output Equivalence:** Only verified tokens consistent with greedy decoding on the final prompt are exposed to the user; unverified drafts or cache states from mismatched prompts are not treated as valid final-prompt outputs.
- **Evaluation Data:** Adapting Daily, Correction, and Entity Tracking families from **Full-Duplex-Bench-v2** in two settings:
  1. Timestamped partial-transcript replay (to isolate scheduling/latency under varying load).
  2. Complete interactive multi-turn dialogues with an external Automated Examiner (to verify conversation quality).
- **Optional Draft Plan:** One possible implementation roadmap is documented in [EXPERIMENT_PLAN.md](file:///home/hice1/nparikh44/scratch/conv-ai-project/EXPERIMENT_PLAN.md) as a starting point, not a requirement.

---

## Environment Notes

- **Cluster:** Georgia Tech PACE cluster. Login nodes do not have GPUs attached; GPU workloads (e.g., A100/H100) are scheduled via Slurm (`sbatch` / `srun`).
- **Workspace:** `/home/hice1/nparikh44/scratch/conv-ai-project`
