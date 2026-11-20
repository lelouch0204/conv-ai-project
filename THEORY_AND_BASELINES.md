# Baselines & Theoretical Latency Model

> Companion reference to [EXPERIMENT_PLAN.md](file:///home/hice1/nparikh44/scratch/conv-ai-project/EXPERIMENT_PLAN.md) and [CAI_Final_Project.pdf](file:///home/hice1/nparikh44/scratch/conv-ai-project/CAI_Final_Project.pdf).

---

## Part 1: Baselines & Method Comparison Framing

In every comparison, the underlying system setup is identical: $N$ concurrent speakers sharing **1 GPU** and **1 frozen LLM**, with the exact same verification rule at the end of each turn so that **every method outputs the exact same final answer text**. The only difference is *when* the GPU allocates capacity to early drafts while speakers are still talking.

### The Three Baselines

1. **Baseline 1 — Never Draft Early (`NoDraft` / "Wait-for-End")**
   - **Behavior:** Ignores partial speech completely. Waits until a speaker stops talking (`utterance_end`), then generates their response from scratch.
   - **Role:** Represents a standard cascaded voice pipeline (0% wasted draft compute, 0 interference from drafts, slowest turn-taking latency when the GPU has spare capacity).

2. **Baseline 2 — Always Draft Early (`FixedInterval` / "Naive PredGen")**
   - **Behavior:** Triggers an early draft every $\Delta T$ ms as new partial words arrive, for every active speaker regardless of server load.
   - **Role:** Represents single-user *PredGen* deployed naively on a shared GPU. Achieves strong latency when 1 speaker is alone with steady speech, but introduces contention and wasted compute when multiple speakers talk concurrently or revise mid-utterance.

3. **Baseline 3 — Speech-Smart, Load-Blind (`SemanticOnly` / Edit-Trigger Only)**
   - **Behavior:** Triggers a draft only when the user's partial transcript appears semantically complete/answerable (or has not just undergone a mid-sentence correction), **without** considering current GPU load or queue backlog.
   - **Role:** Represents speech-side triggering (*LTS-VoiceAgent* style). Tests whether speech signals alone suffice without shared-GPU resource control when multiple speakers finish turns near the same time.

### How the Proposed Controllers Fit

| Method | Uses Speech / Edits? | Uses Shared GPU Load? | What It Isolates |
| :--- | :---: | :---: | :--- |
| **1. `NoDraft` (Wait-for-End)** | No | No | Floor on GPU waste; non-speculative latency baseline |
| **2. `FixedInterval` (Naive PredGen)** | No | No | Shows when unconditional early drafting breaks on a shared GPU (**RQ1**) |
| **3. `SemanticOnly` (Speech-Trigger Only)** | **Yes** | No | Shows whether speech signals alone suffice under contention |
| **4. `LoadReuseAdaptive` (Core Systems Controller)** | No | **Yes** | Prioritizes completed turns & throttles drafting by queue backlog and recent draft reuse |
| **5. `LoadReuse + Speech/Edit`** | **Yes** | **Yes** | Tests whether speech stability / semantic readiness adds value *beyond* load & reuse control (**RQ2**) |

* *(See [EXPERIMENT_PLAN.md](file:///home/hice1/nparikh44/scratch/conv-ai-project/EXPERIMENT_PLAN.md) for the exact 7-row evaluation matrix, including separating `+ Edit` vs. `+ Semantic` and the `ReuseOnly` ablation.)*

- **Comparing (1) vs. (2) vs. (4)** answers whether resource-aware draft scheduling outperforms both never drafting and unconditional drafting.
- **Comparing (3) and (5) vs. (4)** answers whether transcript-edit or semantic-readiness signals improve scheduling once GPU load and recent reuse are already controlled.

---

## Part 2: Theoretical Latency & Gain Model

We can quantify theoretical post-speech latency (Time-To-First-Sentence, TTFS) and expected latency gains using a closed-form GPU roofline model.

### 1. Setup & Variables

Assume a speaker finishes speaking at $t = 0$ (`utterance_end`), and the assistant must validate a **first sentence of $L$ tokens** (e.g., $L \approx 20$ tokens for TTS synthesis) before audio output can begin.

- **$t_{\text{dec}}(B)$:** Time for **1 sequential decode step** at active batch size $B$. Under an affine roofline model:
  $$t_{\text{dec}}(B) = t_0 + c \cdot B$$
  where $t_0$ is the fixed cost of loading model weights from HBM (dominant at small $B$) and $c$ is the marginal per-sequence cost (KV-cache reads + extra compute, which grows once $B$ crosses the memory-bandwidth knee).
- **$t_{\text{prefill}}(n)$:** Time for **1 parallel forward pass** over $n$ tokens. Because the $n$ tokens are processed in parallel matrix multiplications, $t_{\text{prefill}}(K) \ll K \cdot t_{\text{dec}}(B)$.
- **$K$:** Number of candidate tokens drafted while the user was still speaking ($K < L$).
- **$\alpha \in [0, 1]$:** **Draft acceptance rate** (fraction of the $K$ drafted tokens matching greedy decoding on the final prompt).

---

### 2. Post-Speech Latency Equations

#### A. Without Early Drafting (`NoDraft`)
At $t = 0$, the server waits for any queue delay $W_0$, prefills the final transcript chunk ($\Delta P$ tokens), and runs **$L$ sequential decode steps** alongside $B_{\text{fg}}$ concurrent foreground speakers:

$$T_{\text{no-draft}} = W_0 + t_{\text{prefill}}(\Delta P) + L \cdot t_{\text{dec}}(B_{\text{fg}})$$

#### B. With Early Drafting
While the user was speaking ($t < 0$), the server generated $K$ draft tokens. At $t = 0$:
1. The server runs **1 parallel verification pass** over the final transcript chunk + the $K$ candidate draft tokens: $t_{\text{prefill}}(\Delta P + K)$.
2. That verification pass accepts **$\alpha K$ draft tokens** plus **1 free bonus token** from the logits at position $\alpha K + 1$.
3. The server runs sequential decoding only for the remaining $\max(0, L - \alpha K - 1)$ tokens, while concurrent background drafts inflate the batch size by $B_{\text{draft}}$:

$$T_{\text{draft}} = W_{\text{draft}} + t_{\text{prefill}}(\Delta P + K) + \max(0, L - \alpha K - 1) \cdot t_{\text{dec}}(B_{\text{fg}} + B_{\text{draft}})$$

---

### 3. Net Theoretical Latency Gain ($\Delta T$)

Subtracting $T_{\text{draft}}$ from $T_{\text{no-draft}}$ (for $L > \alpha K + 1$, and writing $L_{\text{rem}} = L - \alpha K - 1$) yields the exact latency saved per turn:

$$\Delta T = T_{\text{no-draft}} - T_{\text{draft}} = \underbrace{(\alpha K + 1)\, t_{\text{dec}}(B_{\text{fg}})}_{\text{1. Sequential steps saved}} \;-\; \underbrace{\Big(t_{\text{prefill}}(\Delta P + K) - t_{\text{prefill}}(\Delta P)\Big)}_{\text{2. Marginal cost to verify } K \text{ tokens}} \;-\; \underbrace{L_{\text{rem}} \cdot c \cdot B_{\text{draft}}}_{\text{3. Batch slowdown from concurrent drafts}} \;-\; \underbrace{(W_{\text{draft}} - W_0)}_{\text{4. Extra queue wait}}$$

#### Key Regimes & Implications

1. **Theoretical Best-Case Ceiling (Low Load $B_{\text{draft}} c \approx 0$, Stable Speech $\alpha = 1$, $K = L - 1$):**
   All $L$ tokens of the first sentence are ready immediately after the **single parallel verification pass**:
   $$T_{\min} = t_{\text{prefill}}(\Delta P + L - 1) \approx 1.1 \times t_{\text{dec}}(1)$$
   Post-speech sequential work drops from $L$ forward passes to **1 parallel forward pass** (saving $(L - 1)\cdot t_0$ in sequential weight-loading overhead).

2. **Expected Relative Latency Under Light Load ($\Delta W \approx 0, \, c \approx 0$):**
   When marginal verification overhead $\Delta t_{\text{prefill}}(K)$ is small compared to $L \cdot t_{\text{dec}}$:
   $$\frac{T_{\text{draft}}}{T_{\text{no-draft}}} \approx 1 - \frac{\alpha K}{L}$$
   For example, drafting $K \approx L$ tokens with $\alpha = 0.6$ acceptance yields roughly a $60\%$ post-speech latency reduction on a lightly loaded GPU.

3. **Shared-GPU Break-Even Inequality (When Early Drafting Helps vs. Hurts):**
   Early drafting produces a net latency gain ($\Delta T > 0$) if and only if:
   $$\alpha K + 1 > \frac{\Delta t_{\text{prefill}}(K) + L_{\text{rem}} \cdot c \cdot B_{\text{draft}} + (W_{\text{draft}} - W_0)}{t_{\text{dec}}(B_{\text{fg}})}$$
   where $\Delta t_{\text{prefill}}(K) = t_{\text{prefill}}(\Delta P + K) - t_{\text{prefill}}(\Delta P)$.
   - When **server load rises** ($W_{\text{draft}} - W_0$ and $c \cdot B_{\text{draft}}$ grow) or **speech revisions reduce stability** ($\alpha \to 0$), the right-hand side exceeds the left-hand side ($\Delta T < 0$), meaning early drafting **increases** latency compared to `NoDraft`.
   - This break-even condition directly motivates resource-aware control: the scheduler estimates expected reuse ($\hat{\alpha}$) and server contention ($\Delta W, B$) and restricts drafting to regimes where the inequality holds.

---

## Part 3: Oracle Upper Bounds for Evaluation

To benchmark how close practical controllers get to the theoretical ceiling in experiments, we can include two Oracle reference lines alongside `NoDraft` and the evaluated policies:

1. **Perfect-Prefix Theoretical Oracle ($\alpha = 1$, Zero Wasted Compute):**
   Assumes every drafted token matches the final prompt ($\alpha = 1$, $B_{\text{draft}} = 0$ overhead on foreground), establishing the hardware floor $T_{\min}$ after `utterance_end`.
2. **Hindsight Trace Oracle (Clairvoyant Scheduler):**
   Knows future trace timestamps and revision points in advance; drafts only on the last stable prefix prior to `utterance_end` and only when idle GPU capacity exists such that $\Delta T > 0$.

Reporting **`% of Hindsight Oracle Latency Reduction Captured`** alongside **`% of Wasted Draft Tokens Avoided`** grounds empirical policy gains directly against the theoretical maximum achievable on each workload.
