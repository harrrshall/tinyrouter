# TRINITY Replication — Canonical Implementation Spec (SPEC.md)

> Generated from a multi-agent deep read of the paper, then corrected by an adversarial
> review and grounded against the real Qwen3-0.6B config and a live Fireworks probe.
>
> **Section 0 below is authoritative.** Where any number later in this document disagrees
> with Section 0, Section 0 wins (it folds in the review corrections + verified facts).

---

## 0. VERIFIED FACTS & CORRECTIONS (authoritative — override anything below)

### 0.1 Verified against ground truth (not the paper's prose)
- **Qwen3-0.6B real config** (`huggingface.co/Qwen/Qwen3-0.6B/config.json`):
  `hidden_size = 1024` → **d_h = 1024 CONFIRMED**; `num_hidden_layers = 28` →
  **second-to-last layer = index 26**; `num_attention_heads = 16`, `num_key_value_heads = 8`
  (**GQA**), `head_dim = 128`; `intermediate_size = 3072` (**SwiGLU**: gate/up/down);
  `tie_word_embeddings = True`; `bfloat16`.
  Per-layer linear matrices and their SVD singular-value counts (= min dim):
  `q_proj 1024×2048→1024`, `k_proj 1024×1024→1024`, `v_proj 1024×1024→1024`,
  `o_proj 2048×1024→1024`, `gate_proj 1024×3072→1024`, `up_proj 1024×3072→1024`,
  `down_proj 3072×1024→1024`. **All 7 matrices give 1024 SVs each.**
- **Fireworks pool verified LIVE (HTTP 200, real completions):** `deepseek-v4-pro`, `glm-5p2`,
  `kimi-k2p6` all exist and answer. (The review flagged these as possibly fabricated IDs — that
  concern is resolved by the live probe; they are real serverless models on our account.)
- **Remote box:** 8× H200 NVL (143 GB), GPU **index 5 only**, /mnt/data 3.2 TB free.

### 0.2 Corrected numbers (the review caught arithmetic errors below)
| Quantity | WRONG (in body) | CORRECT (use this) | Why |
|---|---|---|---|
| Pool size L | — | **3** | our setup |
| Head output n_a = L+3 | — | **6** | 3 agents + 3 roles |
| Linear head params | — | **6,144** = 6×1024 | correct |
| **SVF scales** | "9,216 (Q/K/V/O+MLP)" | **7,168** = 7×1024 [OUR CHOICE] | paper's 9,216=9×1024 does NOT map onto Qwen3's 7 linear matrices; SVF-ing all 7 of layer 26 = 7,168. Verify empirically (S2). |
| **Total trainable n** | 15,360 | **13,312** = 6,144 + 7,168 | head + SVF |
| **CMA population λ** | "34" | **33** = ⌈4 + 3·ln(13312)⌉ = ⌈32.49⌉ | review caught 34 is wrong (also 33 for n=15,360) |
| Parents μ | 17 | **16** = ⌊33/2⌋ | follows λ |
| **Budget B_env** | "34,560" | **31,680** = 16·33·60 | 16·34·60≠34,560 anyway; product error |
| REINFORCE batch = m·λ | "544" | **528** = 16·33 | followed wrong λ |

### 0.3 Gaps the review exposed — decisions now fixed (were silently assumed)
1. **SVF matrix set & count:** apply SVF to **all 7 linear matrices of layer 26** → 7,168 learnable
   singular-value scales, all initialized to **1.0** (identity ⇒ unmodified SLM at start). The
   smoke test **S2 must print the actual count** and assert it equals what we pack into θ; do not
   trust 7,168 blindly until S2 confirms it on the loaded checkpoint.
2. **Hidden-state norm / σ₀ coupling:** raw bf16 penultimate hidden states can have large norm, so
   `σ₀=0.1` on a `W=0` start may saturate softmax. **OUR CHOICE: L2-normalize `h` (h ← h/‖h‖)
   before the head**, making logit scale ‖h‖-independent and `σ₀` well-behaved. S1 prints `‖h‖`
   to confirm. (Minor deviation from paper; documented in JOURNAL.)
3. **MT-Bench reward:** MT-Bench is **held-out / report-only** (10-pt LLM-judge score, kept out of
   any Bernoulli training reward). We never binarize it for fitness. R2's "every held-out task"
   check uses the 10-pt score as a ranking, not a {0,1} reward.
4. **Budget-matched single-model baselines (R1/R2):** run each single model at **max_tokens =
   20,480 (5×)** so the single-vs-TRINITY comparison is fair, matching the paper's 5× protocol.
5. **Verifier turn-1 guard:** a Verifier ACCEPT terminates **only if ≥1 Worker output already
   exists** in the transcript; otherwise treat ACCEPT as REVISE. Prevents a turn-1 Verifier from
   accepting an empty solution (guaranteed reward-0 trajectories that bias CMA against Verifier).
6. **Reward checkers are first-class, not "OUR CHOICE niceties":** they define the fitness signal.
   `reward.py` must implement: code pass@1 via a sandboxed executor with timeout; math via
   boxed-answer / last-number extraction + symbolic equality; MMLU/GPQA via robust letter
   extraction. Each gets a unit test (S5) with one known-correct + one known-wrong case.
7. **Caching × greedy × binary-reward interaction:** disk cache is keyed by
   `(model, prompt_hash, decode_params)`. With temp 0.0 this makes trajectories deterministic, so
   two CMA candidates that pick the same (agent, role) sequence on the same instance get identical
   rewards. This is acceptable (reduces cost) but means **inter-candidate fitness variance comes
   only from different (agent, role) choices** — monitor that candidates actually diverge.

### 0.4 CMA constants — which `n`?
The block-ε-separability that justifies sep-CMA-ES was measured on the paper's 7-agent
representation. We **cannot assume it transfers** to a 3-model pool, so **R8 (CMA > SFT > RS >
REINFORCE) is a hypothesis to test, not a given.** We compute λ on our actual joint n = 13,312 and
use the `cma` library's separable defaults for the rest. If S7/early iterations show CMA not
improving, the first thing to re-examine is the separability assumption on our pool (log to JOURNAL).

---
---

# TRINITY Replication — Canonical Implementation Spec (SPEC.md)

> Re-implementation of **TRINITY: An Evolved LLM Coordinator** (ICLR 2026, arXiv:2512.04695v3) for OUR setting: a **3-model Fireworks pool** coordinated by a **local Qwen3-0.6B** SLM, trained with **sep-CMA-ES** on a **single H200 (GPU index 5, ~143 GB)**.
>
> Source paper: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt`.
> Every paper-quoted number is preserved. Where the paper is silent, a default is proposed and tagged **[OUR CHOICE]**. Where OUR setting deviates by design (pool size), it is tagged **[REPLICATION DELTA]**.

---

## 1. Goal & Success Criteria

### 1.1 What TRINITY is
A tiny (**< 20K trainable params**) coordinator that, at each of up to **K=5** turns, reads the full conversation transcript with a frozen-ish **Qwen3-0.6B** SLM, and via a **~10K-param linear head** emits two decisions: **which LLM** to call and **which role** (Thinker / Worker / Verifier) it plays. The coordinator's own generated text is discarded; only the head logits matter. It is trained derivative-free with **sep-CMA-ES** against a **binary terminal reward** (task correct = 1).

### 1.2 OUR setup (fixed)
- **Coordinator SLM:** Qwen3-0.6B, run **locally on H200 GPU5**. Hidden dim `d_h = 1024`.
- **Coordinated pool (L = 3) [REPLICATION DELTA]:** Fireworks-served `deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6`. (Paper used L=7: GPT-5, Gemini-2.5-pro, Claude-Sonnet-4-20250514, Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B reasoning, Qwen3-32B direct.)
- **Head output `n_a = L + 3 = 6`** logits (3 agent + 3 role). Linear head = `6 × 1024 = 6,144` params (paper: 10,240 at L=7).
- **Roles:** Thinker (T), Worker (W), Verifier (V).

### 1.3 Success criteria — RELATIVE, not absolute
Absolute scores will differ (different pool, no GPT-5/Gemini/Claude). We replicate the paper's **relative invariants** (paper anchors in parentheses):

| # | Invariant to reproduce | Paper evidence |
|---|---|---|
| R1 | TRINITY avg > best single model avg (budget-matched 5×) | 70.44 > GPT-5 65.95 (in-dist); 54.21 > Gemini 52.34 (held-out) |
| R2 | TRINITY > every single model on every task | Tables 1, 2 |
| R3 | TRINITY > best multi-agent baseline (MoA/MasRouter/RouterDC/Smoothie) | §4.2, Fig.3 |
| R4 | TRINITY > random routing | RLPR: 0.41 vs 0.32 |
| R5 | TRINITY ≈ Per-Question-Best on 3 of 4 in-dist tasks | §4.2–4.3 |
| R6 | Lifting token cap → large LiveCodeBench jump, beats all constituents | 0.61 → 0.862, beats GPT-5 0.838 |
| R7 | More max-turns → monotonic gain | 0.823→0.863 (2→6 turns) |
| R8 | sep-CMA-ES > SFT > RS > REINFORCE on all 4 tasks | Table 4 |
| R9 | Removing SVF / Thinker / tri-role / penultimate-token all hurt; tri-role + token-choice matter most | Table 2 |
| R10 | linear head ≥ all other head variants overall | Table 3 |
| R11 | Trained coordinator > LLM-as-coordinator | Table 8 |
| R12 | TRINITY far more token-efficient than MoA/Smoothie/MasRouter | Table 9 |
| R13 | Mean relative-error-reduction ≈ 21.9% vs 2nd-best (ballpark, pool-dependent) | §1 |

**Definition of done for the replication:** R1–R4 and R8 hold on at least 2 of our chosen in-distribution tasks; the trained coordinator runs end-to-end within the atomic-eval budget on one H200; the optimizer drives `J(θ)` upward over iterations.

---

## 2. System Architecture + Data-Flow Diagram

Two nested loops:
- **Inner (coordination run):** one trajectory `τ` = one atomic Bernoulli evaluation. Up to K turns of (select agent+role → prompt LLM → post-process → append).
- **Outer (training):** sep-CMA-ES samples candidate θ vectors, evaluates each by averaging `m_CMA` inner runs, recombines into the next parent.

```
                          OUTER LOOP  (sep-CMA-ES, ~60 iters)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  parent θ (mean m_t, step σ_t, diag D_t)                                   │
  │     │ sample λ candidates  y = m_t + σ_t·D_t·z,  z~N(0,I)                   │
  │     ▼                                                                      │
  │  for each candidate θ_i:  fitness = mean over m_CMA inner runs of R(τ)     │
  │     │                                                                      │
  │     ▼  fitness-weighted recombination → new m_{t+1}, σ_{t+1}, D_{t+1}      │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │ θ defines (head W) + (SVF singular-value scales)
                                    ▼
   INNER LOOP  (one trajectory τ = 1 Bernoulli atomic eval)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ user query Q ──► transcript C_0 = [Q]                                    │
   │                                                                          │
   │ for k = 1..K:                                                            │
   │   ┌───────────────────────────────────────────────────────────┐        │
   │   │ s = concat(C_{k-1})  (Q + all prior outputs O_1..O_{k-1})   │        │
   │   │            │                                                │        │
   │   │            ▼                                                │        │
   │   │   Qwen3-0.6B (SVF-adapted, FROZEN orthogonal factors)       │        │
   │   │   forward over [..<Head Input><EOS>]                        │        │
   │   │            │  h = hidden state @ PENULTIMATE output token   │        │
   │   │            ▼  (final layer, R^1024)                         │        │
   │   │   LINEAR HEAD  z = W·h     W∈R^{6×1024}                      │        │
   │   │            │                                                │        │
   │   │     ┌──────┴───────┐                                        │        │
   │   │  agent logits[0:3] role logits[3:6]                         │        │
   │   │     │softmax        │softmax                                │        │
   │   │     ▼               ▼                                       │        │
   │   │   A_k ∈ pool      R_k ∈ {T,W,V}                             │        │
   │   └───────────────────────────┬───────────────────────────────┘        │
   │                               ▼                                          │
   │           MESSAGE-PROCESSING MODULE: inject role-specific prompt(R_k,C)  │
   │                               │                                          │
   │                               ▼                                          │
   │             Fireworks LLM A_k .generate(prompt, max_tokens=4096)  → M_k  │
   │                               │                                          │
   │                               ▼                                          │
   │             post-process M_k → O_k ; C_k = C_{k-1} ∪ {O_k}               │
   │                               │                                          │
   │     if R_k == V and parse(O_k)==ACCEPT:  τ=k; break                      │
   │   end for  (else τ=K)                                                    │
   │                               ▼                                          │
   │             final answer = O_τ ──► reward R(τ)∈{0,1} (correctness)       │
   └────────────────────────────────────────────────────────────────────────┘
```

**Module mapping:** SLM+head = `src/trinity/coordinator`; pool = `src/trinity/llm`; role prompts/post-processing/verifier-parse = `src/trinity/roles`; inner loop = `src/trinity/orchestration`; outer loop = `src/trinity/optim`.

---

## 3. Coordinator: Qwen3-0.6B + Hidden-State Extraction + ~10K Head

### 3.1 SLM
- **Model:** Qwen3-0.6B (paper §4.1). `d_h = 1024` (penultimate-layer hidden size; verify against the actual checkpoint config at load — paper asserts 1024).
- Run on **H200 GPU5** in bf16/fp16.

### 3.2 Hidden-state extraction (exact)
- **Input construction:** feed the full transcript as the SLM input. Figure 2 shows a constructed sequence `<BOS> ... <Head Input> <EOS>`. The head reads the **penultimate output token** position (the `<Head Input>` position) at the **final hidden layer**.
- **Token:** penultimate output token — NOT the last/EOS token. (Ablation: using last token collapses LiveCodeBench by >10 points, 61.46→50.85.)
- **Pooling:** none. Single token's vector `h ∈ R^{1024}` is the head's **sole input**. No turn index, no role one-hots, no history features.
- The SLM's generated text is discarded; the design even permits using an earlier-than-penultimate token for speed, but **all experiments use penultimate** — so do we.

**[OUR CHOICE] Exact mechanics:** Append a fixed prompt suffix to the transcript that asks the SLM to "decide the next agent and role," run a 1-token forward (or generate up to a `<think>`/EOS so a penultimate position exists), and take the hidden state at the second-to-last position of the produced sequence. Standardize on: tokenize `transcript + "\n<Head Input>"`, do one forward pass with a single appended EOS, read index `-2`. Document this as the canonical extraction so train and inference match.

### 3.3 Head (linear, default)
- **Equation 5:** `z = W·h`, `W ∈ R^{n_a × d_h}`, **no bias, no activation**.
- **Output size `n_a = L + 3`.**
  - Paper: L=7 → n_a=10 → `W ∈ R^{10×1024}` = **10,240 params**.
  - **[REPLICATION DELTA] OURS:** L=3 → **n_a = 6** → `W ∈ R^{6×1024}` = **6,144 params**.
- **Decision mapping:** split logits → first `L` = agent logits, last `3` = role logits. Apply **softmax separately** to each group (two independent categoricals); the policy is `π_θ(a|s) ∝ exp(f_θ(h)_a)`.
- **No stop logit.** Termination is implicit (Verifier selected + ACCEPT, or K reached).

### 3.4 The trainable parameter vector θ for CMA-ES
θ = concatenation of two sets (total **< 20K**):

| Component | Paper (L=7) | OURS (L=3) [REPLICATION DELTA] |
|---|---|---|
| Linear head `W` | 10,240 (`10×1024`) | **6,144** (`6×1024`) |
| SVF singular-value scales (Qwen3-0.6B 2nd-to-last layer) | 9,216 | 9,216 (unchanged) |
| **Total dim `n`** | **19,456** | **15,360** |

- **SVF (Transformer² / Sun et al. 2025):** SVD a selected subset of weight matrices in the **second-to-last layer** of Qwen3-0.6B; **learn only singular-value scales**, keep U, V (orthogonal factors) **frozen**. 9,216 scales.
- **[OUR CHOICE] Which matrices get SVF:** paper does not enumerate. 9,216 = `9×1024`. Default: apply SVF to the Q,K,V,O attention projections + gate/up/down MLP of the 2nd-to-last block, taking the top singular components so the learnable-scale count is 9,216; initialize all scales to **1.0** (identity). If the exact matrix set can't hit 9,216, fall to "all linear weight matrices in that block, scales=1.0" and record the actual count.

**[OUR CHOICE] Optimize head + SVF jointly** (single θ of dim 15,360). The paper's "<20K" total and Fig.2 both imply joint optimization; theory's `n≈10000` refers loosely to the head.

### 3.5 Alternative heads (ablation parity, Appendix A.4)
All map `h∈R^{d_h} → z∈R^{n_a}`. Param counts shown at paper's `d_h=1024, n_a=10`:

| Head | Equation | Params (n_a=10) | Notes |
|---|---|---|---|
| **linear** (default) | `z=Wh`, no bias | 10,240 | best overall |
| low-rank | `u=ELU(Uh, α=0.1); z=Vu·σ`; `r=14`, σ fixed; Xavier-uniform init | 20,680 | gains `U~U[±√(6/(d_h+r))]`, `V~U[±√(18/(r+n_a))]` |
| sparse | `z=W(h⊙α)`; `k=max(1,⌊d_h(1−σ(ρ))⌋)`; Gumbel top-k τ∈[1.0,20.0]; hard top-k at inference | 11,266 | `d_h·n_a + d_h + 2` |
| block-diagonal-2 | B=2 proportional blocks | 5,120 | |
| block-diagonal-10 | B=10, one block/logit, **argmax** output | 1,024 | exact 10× reduction |

---

## 4. Roles + Multi-Turn Protocol + Prompt Templates

### 4.1 Role contracts (verbatim §3.2)
- **Thinker strategizes:** returns meta-level guidance — high-level plans, decompositions, critiques of partial solutions; may propose subgoal plan; may specify the role of the next agent.
- **Worker executes:** acts directly on the task; produces actionable content (derivation, code snippet, numerical result).
- **Verifier evaluates:** checks if the accumulated solution is correct/complete/responsive to Q; outputs `u_k ∈ {ACCEPT, REVISE}` + optional diagnosis `δ_k`. If ACCEPT → signal termination.

### 4.2 Per-turn protocol
1. Build `C_{k-1} = (Q, O_1, …, O_{k-1})` — query concatenated with all prior outputs.
2. SLM forward → `h` → head → softmax → pick `A_k` (agent) and `R_k` (role).
3. Message-processing module injects role-specific prompt over `C_{k-1}`.
4. Query `A_k` → message `M_k` (max 4096 tokens, minimal reasoning effort).
5. Post-process `M_k` → `O_k`; append to transcript.
6. If `R_k = V` and `parse(O_k) = ACCEPT` → terminate.

**Termination:** `τ = min{ k ≤ K : R_k = V ∧ u_k = ACCEPT }`, else `τ = K`. Final answer = `O_τ`.
**Horizon:** `K = B_turn = 5` (treat as identical).
**Decoding (global, all pool LLMs):** `max_tokens = 4096`, "minimal reasoning effort". No temperature/top-p stated.

### 4.3 Inference selection rule
**[OUR CHOICE]** Use **argmax** over each softmax group at deployment/eval for reproducibility; use **sampling** from `π_θ` only inside training fitness evaluation so the optimizer sees the stochastic policy it is optimizing. (Paper: default conversion is softmax; block-diag-10 uses argmax; sampled-vs-argmax at inference is unspecified for the linear head.)

### 4.4 Prompt templates — NOT IN PAPER, must author [OUR CHOICE]
The paper gives only contracts + one Fig.1 example ("The next agent should act as a solver…"). No verbatim templates exist. Author these system prompts:

**THINKER**
```
You are the THINKER. Do NOT solve the task end-to-end.
Analyze the current state and produce meta-level guidance: a concise high-level
plan, a decomposition into subgoals, or a critique of the partial solution so far.
You may recommend which role should act next.
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
Return only your plan/critique.
```

**WORKER**
```
You are the WORKER. Make concrete progress toward the final answer.
Follow any plan in the transcript. Produce actionable content: the derivation,
the code, or the numerical/final result. Be explicit and complete.
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
Return your solution work.
```

**VERIFIER**
```
You are the VERIFIER. Check whether the accumulated solution is correct, complete,
and responsive to the query.
End your response with EXACTLY one line:
VERDICT: ACCEPT      (if the current answer is correct and final)
VERDICT: REVISE      (otherwise, with a one-line diagnosis above it)
QUERY:
{Q}
TRANSCRIPT SO FAR:
{C_prev}
```

### 4.5 Post-processing `M_k → O_k` — NOT IN PAPER [OUR CHOICE]
Paper says only "lightly post-processes / condenses / extracts." Default: **pass-through with light truncation** — keep `M_k` verbatim but cap each `O_k` at a fixed char/token budget (e.g. 2,000 tokens) to bound transcript growth; for Verifier turns, store `(u_k, δ_k)` where `δ_k` is the text above the VERDICT line. Do NOT add an extra summarizer LLM call (keeps atomic-eval cost predictable). Revisit only if transcripts overflow the SLM context.

### 4.6 Verifier ACCEPT/REVISE parsing — NOT IN PAPER [OUR CHOICE]
Parse the **last** `VERDICT: <ACCEPT|REVISE>` line (case-insensitive, regex `VERDICT:\s*(ACCEPT|REVISE)`). If absent, default to **REVISE** (fail-safe: never terminate on an unparseable verifier). This makes the ACCEPT signal deterministic and decouples it from free-text.

### 4.7 First-turn behavior [OUR CHOICE]
Paper does not force a first role. **Let the head choose freely** from turn 1 (the formal rule permits even a turn-1 Verifier-ACCEPT). No forced T→W→V ordering.

---

## 5. sep-CMA-ES (Optimizer)

### 5.1 Search space
θ ∈ ℝ^n, n = head + SVF (joint). OURS: **n = 15,360**. Paper theory: `n ≈ 10000`.

### 5.2 Fitness
- **Atomic evaluation** = one full trajectory τ → Bernoulli reward `R(τ) ∈ {0,1}`.
- **Fitness(θ) = mean of `R(τ)` over `m_CMA` replications.** Objective `J(θ) = E_{τ~π_θ}[R(τ)]`, **maximized**.
- **No explicit cost/length penalty** in fitness; cost is bounded structurally by `max_tokens=4096` × `K=5`.
- `B_env` counts individual Bernoulli calls. CMA cost/iteration = `m_CMA · λ`; `T = ⌊B_env / (m_CMA·λ)⌋`.

**[OUR CHOICE] Replication construction:** each of the `m_CMA` draws uses a **different randomly-sampled task instance** from the training set (a minibatch of 16 distinct problems per candidate), re-sampled per iteration. (Paper is ambiguous: SFT used 3 seeds, RS used 32 trials; "replication/averaging" unspecified.) This gives an unbiased low-variance estimate of `J`.

### 5.3 Hyperparameters

| Param | Paper value (n≈10000) | OURS (n=15,360) | Source / note |
|---|---|---|---|
| Algorithm | sep-CMA-ES (diagonal covariance, Ros & Hansen 2008) | same | §3.3 |
| Population `λ` | `⌈4 + 3 ln n⌉ = 32` | **`⌈4+3 ln 15360⌉ = 34`** [REPLICATION DELTA] | recompute on our n |
| Replication `m_CMA` | 16 | 16 | §3.3 |
| Parents `μ` | symbolic only | **`⌊λ/2⌋ = 17`** [OUR CHOICE] | sep-CMA-ES default |
| Recombination weights `w_j` | symbolic only | **default log-weights** `w_j ∝ ln(μ+0.5) − ln j`, normalized [OUR CHOICE] | Ros & Hansen default |
| Diag cov learning rate `c_cov` | `Θ(1/n)` | use library default `(n+2)/3` form ≈ Θ(1/n) | §A.1 |
| `c_σ`, `d_σ`, `c_1`, `c_μ` | not given | **library defaults** [OUR CHOICE] | use pycma / sep variant |
| Initial mean `m_0` | not given | **head W = 0 (uniform policy); SVF scales = 1.0** [OUR CHOICE] | symmetric start |
| Initial step `σ_0` | not given | **0.1** [OUR CHOICE] | small, since W=0 start; tune if collapse |
| Bounds on θ | none stated (RS-only band) | **none** (free) [OUR CHOICE] | clip only if divergence |
| Restart logic | none | **none** (single run) | matches paper |
| Iterations `T` | analyzed [2,60]; 60 used for budget-match | **target ~60** [OUR CHOICE] | |
| Total budget `B_env` | 1.5k–40k | `≈ m_CMA·λ·T = 16·34·60 = 34,560` (within range) | |
| Output conversion | softmax (argmax for block-diag-10) | softmax (train), argmax (eval) | §Table 3 caption |

### 5.4 Why this optimizer (claimed advantage)
High dim + weak parameter coupling + high per-step cost → REINFORCE per-parameter gradients are low-SNR. The objective exhibits **block-ε-separability** (Def. 1: scaled Hessian is near block-diagonal), so a diagonal-covariance ES is well-matched. Theory (Prop 1) gives CMA/RS gain ratio ≈ `(T/ln(16T))·η²` > 1 even for small T; (Prop 2) per-iteration contraction `~(κ̄_{μ,λ}/n)(1−O(ε_H))` after a Θ(n) transient.

---

## 6. Benchmarks & Eval Protocol

### 6.1 In-distribution (train + eval), four tasks
MATH500, MMLU, RLPR, LiveCodeBench. **Train per-task, eval on the matching test set** (one coordinator per benchmark — no multi-task blend [OUR CHOICE matches paper]).
- **LiveCodeBench (exact split):** train on **V1 (400 samples)**, eval on **V6 newly-introduced (175 samples)**, Jan–Apr 2025.
- Combined training pool ≈ **7,000 datapoints**; SFT oracle used **3 seeds**.
- **[OUR CHOICE] Split sizes for MATH500/MMLU/RLPR** (paper says "official splits where available" but gives no numbers): use official train/test; if none, hold out a fixed 20% as test, seed=0. Document actual sizes used.

### 6.2 Held-out / zero-shot transfer (no retraining)
AIME2025, BigCodeBench, MT-Bench(-101), GPQA-Diamond. Same K=5 / 4096-token settings assumed.

### 6.3 Metrics
- LiveCodeBench: **pass@1** (execute tests).
- MATH500 / GPQA-D / AIME: exact answer-match accuracy.
- MMLU / BigCodeBench: accuracy / pass@1.
- MT-Bench: ~10-point LLM-judge score (**[OUR CHOICE]** judge = strongest pool model, GPT-4-class rubric; keep separate from accuracy averages).
- Reward `R(τ) ∈ {0,1}` per atomic eval (correctness checker per task above).
- **Relative Error Reduction:** `RER = (Z − S*)/(1 − S*)` where Z = coordinated score, S* = best single-agent on subset.
- **Per-Question-Best** upper bound = union of correct answers across the pool.

### 6.4 Seeds & decoding
- **[OUR CHOICE] 3 eval seeds** (matches paper's only stated seed count), report mean ± std.
- **[OUR CHOICE] Decoding for pool LLMs:** temperature **0.0** (greedy) for deterministic correctness scoring on math/code; `top_p=1.0`. "Minimal reasoning effort" → set each Fireworks model's reasoning/thinking budget to its lowest documented setting. Document the exact per-model mapping.

### 6.5 Full-power LiveCodeBench
After training, **lift the 4096-token cap, no retraining** (R6). Paper's pool narrowed to 3 closed models; OUR pool is already 3 models, so simply remove the cap and re-eval.

---

## 7. Target / Relative Results to Reproduce

### 7.1 Paper anchor numbers (for sanity, not absolute targets)
**In-distribution (linear head, 4096 cap, K=5):** TRINITY = LiveCodeBench **61.46** (0.615) / MATH500 **88.00** / MMLU **91.56** / RLPR **40.72**; **Avg 70.44**.
**Held-out (Table 1):** AIME 50.00 / BigCodeBench 35.80 / MT-Bench 9.60 / GPQA-D 76.82; **Avg 54.21** (beats best single Gemini 52.34).
**Full-power LCB:** **0.862 ± 0.005** (GPT-5 0.838, Gemini 0.672, Claude 0.465); turns 2→6: 0.823→0.863.

**Optimizer comparison (Table 4):**
| | LCB | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| REINFORCE | 0.253 | 0.459 | 0.500 | 0.266 |
| RS | 0.374 | 0.794 | 0.897 | 0.345 |
| SFT | 0.592 | 0.786 | 0.906 | 0.360 |
| **sep-CMA-ES** | **0.615** | **0.880** | **0.916** | **0.401** |

**Ablations (Table 2, Avg):** full 70.44; w/o SVF 67.85; w/o Thinker 68.69; w/o tri-role 67.02; last-token 64.66; GPT-5 only 65.95; Gemini only 62.99; Claude only 61.12.

**Heads (Table 3):** linear best overall (0.615/0.880/0.916/0.401); sparse edges MMLU only (0.917); block-diag-10 retains much at 1,024 params.

**Token efficiency (Table 9, avg out-tokens):** TRINITY 2,853/1,200/2,141/1,999 ≪ MoA 6,871/5,218/11,086/21,634; Smoothie similar-high; RouterDC lower but worse accuracy.

### 7.2 OUR targets
We do **not** target absolute numbers. We target the **R1–R13 invariants** (Section 1.3). Known paper inconsistencies to ignore: LLM-as-coordinator avg "64.14" (text) vs 53.76 (Table 8) — use **53.76**; LCB rounding 61.46/0.615/0.61/61.49 — same run, target ≈ 61.5.

---

## 8. Module-by-Module Build Plan (`src/trinity/...`)

### `src/trinity/llm`
- `pool.py`: registry of the 3 Fireworks models (`deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6`) with agent IDs A0/A1/A2.
- `client.py`: async Fireworks chat client; `generate(prompt, max_tokens=4096, temperature=0.0, reasoning="minimal")`. Retry/backoff. **[OUR CHOICE]** optional disk response cache keyed by (model, prompt-hash, decode-params) — paper never mentions caching; add it to cut atomic-eval cost since training repeats instances.
- Maps "minimal reasoning effort" to each provider's knob.

### `src/trinity/coordinator`
- `slm.py`: load Qwen3-0.6B on cuda:5; forward pass returning final-layer hidden states; extract **penultimate-token** vector (`h∈R^1024`). Implements §3.2 extraction contract.
- `svf.py`: SVD the 2nd-to-last layer's selected matrices once at init; expose 9,216 learnable scales; reconstruct `W' = U diag(s∘scale) Vᵀ`; load scales from θ.
- `head.py`: linear head `W∈R^{6×1024}` (and the 4 alt heads for ablations); `forward(h) → (agent_logits[0:3], role_logits[3:6])`; softmax/argmax converters.
- `params.py`: pack/unpack θ ↔ (head W, SVF scales); report `n=15,360`.

### `src/trinity/roles`
- `prompts.py`: Thinker/Worker/Verifier templates (§4.4).
- `postprocess.py`: `M_k → O_k` light-truncation (§4.5).
- `verifier.py`: ACCEPT/REVISE regex parser, fail-safe REVISE (§4.6).

### `src/trinity/orchestration`
- `run.py`: `trinity_run(Q, theta) → O_τ` — the inner loop (Section 2 diagram), enforces K=5 and termination rule.
- `reward.py`: per-task correctness checkers (pass@1 executor for code, exact-match for math, choice-match for MMLU/GPQA, judge for MT-Bench) → `R(τ)∈{0,1}`.
- `dataset.py`: loaders + splits (incl. LCB V1/V6); minibatch sampler for `m_CMA` replications.

### `src/trinity/optim`
- `cmaes.py`: sep-CMA-ES wrapper (diagonal covariance), λ=34, μ=17, m_CMA=16, σ0=0.1, init mean (W=0, SVF=1.0), library defaults for the rest; iteration loop to T≈60; logs `J(θ)` per iteration.
- `fitness.py`: evaluate candidate = mean `R(τ)` over 16 sampled instances (parallelized across the population × replication via async LLM calls).
- `baselines.py`: REINFORCE (batch=`m_CMA·λ`, 60 iters), RS (U[−0.5,0.5], 32 trials/candidate, budget-matched), SFT (Adam, lr 1e-6, batch 64, frozen SLM, head-only) for R8.

### Top-level
- `train.py` (per-task training entrypoint), `eval.py` (in-dist + held-out, 3 seeds), `config.yaml` (consolidated hyperparameters from Section 9).

---

## 9. Consolidated Hyperparameter Table

| Param | Value | OURS / note |
|---|---|---|
| Coordinator SLM | Qwen3-0.6B (local, H200 GPU5) | |
| Hidden dim `d_h` | 1024 | verify at load |
| Default head | linear, `z=Wh`, no bias | |
| Head output `n_a = L+3` | 10 (paper) | **6 (OURS)** |
| Linear head params | 10,240 (paper) | **6,144 (OURS)** |
| SVF target | 2nd-to-last layer, learn singular-value scales | matrices: [OUR CHOICE] Q/K/V/O+MLP, scales init 1.0 |
| SVF params | 9,216 | |
| Total trainable `n` | 19,456 (paper) | **15,360 (OURS)** |
| Hidden-state token | penultimate output token, final layer | |
| Pooling | none | |
| Pool size `L` | 7 (paper) | **3 (OURS)** |
| Roles | T / W / V | |
| Max turns `K = B_turn` | 5 | |
| Max gen tokens / LLM | 4096, minimal reasoning effort | full-power: cap lifted |
| Pool decode temp / top_p | not given | **0.0 / 1.0 [OUR CHOICE]** |
| Output conversion | softmax (default), argmax (block-diag-10) | train=softmax-sample, eval=argmax [OUR CHOICE] |
| Reward | binary terminal `R(τ)∈{0,1}` | |
| Optimizer | sep-CMA-ES (diagonal cov) | |
| Population `λ` | 32 (`⌈4+3 ln n⌉`) | **34 (OURS, recomputed)** |
| Parents `μ` | not given | **17 = ⌊λ/2⌋ [OUR CHOICE]** |
| Recombination weights | not given | **default log-weights [OUR CHOICE]** |
| `m_CMA` (replications) | 16 | minibatch of 16 distinct instances [OUR CHOICE] |
| `m_RS` | 32 | RS baseline |
| `c_cov` | Θ(1/n) | library default |
| Init mean `m_0` | not given | **W=0, SVF=1.0 [OUR CHOICE]** |
| Init step `σ_0` | not given | **0.1 [OUR CHOICE]** |
| Bounds / restarts | none | none |
| Iterations `T` | [2,60]; 60 used | **~60 [OUR CHOICE]** |
| Budget `B_env` | 1.5k–40k | ≈34,560 (16·34·60) |
| REINFORCE | batch=`m_CMA·λ`=512(paper)/544(ours), 60 iters; LR/baseline/entropy not given | [OUR CHOICE] defaults |
| RS | U[−0.5,0.5], 32 trials/candidate, budget-matched | |
| SFT | Adam, lr 1e-6, batch 64, frozen SLM, head-only | |
| Low-rank head | r=14, ELU α=0.1, σ fixed, Xavier-uniform init | ablation |
| Sparse head | Gumbel τ∈[1.0,20.0], hard top-k at inference | ablation |
| Eval seeds | 3 (matches paper) | [OUR CHOICE] |
| LCB split | train V1=400, eval V6=175 | |

---

## 10. Known Unknowns & Decisions (each with a recommended default)

| # | Unknown (paper silent/ambiguous) | Recommended default [OUR CHOICE] |
|---|---|---|
| 1 | **Role prompt templates** — none in paper | Templates in §4.4; iterate if a role misbehaves |
| 2 | **`M_k → O_k` post-processing** | Pass-through + token-cap truncation (§4.5); no extra LLM summarizer |
| 3 | **ACCEPT/REVISE parsing** | Require `VERDICT: ACCEPT|REVISE` line; regex parse last occurrence; default REVISE if missing (§4.6) |
| 4 | **Penultimate-token mechanics** | Append `<Head Input>`+EOS, single forward, take index −2; identical at train+eval (§3.2) |
| 5 | **SVF matrix subset** | Q/K/V/O + MLP of 2nd-to-last block, top components → 9,216 scales, init 1.0 (§3.4) |
| 6 | **Joint vs head-only optimization** | Joint (n=15,360) |
| 7 | **Inference selection: sample vs argmax** | Argmax at eval; sample at train fitness (§4.3) |
| 8 | **Agent/role logit factorization** | Two independent softmaxes (separate categoricals) |
| 9 | **CMA `σ_0`, `μ`, weights, `m_0`** | σ0=0.1, μ=17, default log-weights, m_0=(W=0, SVF=1.0) |
| 10 | **Per-task `B_env` / iterations** | T≈60 (B_env≈34,560) for all tasks; tune per-task if budget-bound |
| 11 | **Replication minibatch construction** | 16 distinct random instances per candidate, re-sampled each iteration |
| 12 | **Decode temp/top_p, "minimal reasoning"** | temp 0.0, top_p 1.0, lowest reasoning budget per Fireworks model |
| 13 | **MT-Bench judge & rubric** | Strongest pool model as judge, 10-point; keep separate from accuracy avgs |
| 14 | **MATH500/MMLU/RLPR split sizes** | Official splits; else 80/20 with seed 0; record actual sizes |
| 15 | **Eval seed count** | 3 seeds, report mean±std |
| 16 | **LLM response caching** | Enable disk cache keyed by (model, prompt, decode-params) |
| 17 | **First-turn role** | Free (no forced ordering); turn-1 Verifier-ACCEPT allowed |
| 18 | **`d_h` = 1024 assumption** | Verify against the loaded Qwen3-0.6B config; if it differs, recompute head dims and `n` |
| 19 | **Linear head init** | W = 0 (uniform start), folded into CMA `m_0` |
| 20 | **REINFORCE LR/baseline/entropy** | Library defaults; only batch (=`m_CMA·λ`) and 60 iters are fixed |
---

## 11. Smoke-test ladder = canonical build order (do NOT spend CMA budget until S1–S8 pass)

Build and verify in this order. Each rung is cheap and gates the next; the expensive 60-iteration
CMA run is justified only after S8 returns a finite fitness. (From the adversarial review §E.)

| # | Gate | What it proves | Cost |
|---|---|---|---|
| **S1** | Load Qwen3-0.6B on cuda (GPU5); assert `config.hidden_size==1024`, capture `num_hidden_layers==28`. Run §3.2 extraction on one fixed transcript **twice** → identical `h`, `h.shape==(1024,)`, index −2 is the intended position (not EOS). Print `‖h‖`. | SLM forward + deterministic penultimate extraction; calibrates σ₀. | 1 GPU load |
| **S2** | Build SVF on layer 26; with all scales=1.0 assert reconstructed `W' ≈ W` (max abs diff < 1e-3 bf16) for every targeted matrix; **print the learnable-scale count and assert it == what we pack into θ** (expect 7,168 — verify, don't assume). Perturb one scale → SLM output changes. | SVF round-trips to identity; resolves the #1 blocker. | GPU only |
| **S3** | `unpack(pack(W, svf_scales)) == (W, svf_scales)`; assert `len(θ) == 6144 + actual_svf_count` and that this equals the `n` fed to CMA. | θ layout integrity. | trivial |
| **S4** | Mock the LLM: stub Worker text + a Verifier emitting `VERDICT: ACCEPT`. Run `trinity_run(Q, θ_random)`: assert K≤5, transcript grows, ACCEPT terminates (only after a Worker exists — §0.3.5), returns `O_τ`. Then a no-VERDICT verifier → fail-safe REVISE → runs to K. | Inner loop, termination rule, verifier parser. | $0 |
| **S5** | Feed each `reward.py` checker one known-correct + one known-wrong case (math boxed answer, code pass@1 on a 2-test toy, MMLU letter). Assert {1,0}. | Reward signal correctness (where silent failures hide). | $0 |
| **S6** | One live call to each of the 3 Fireworks models: "minimal reasoning" param accepted, `max_tokens` honored, cache write/read works. | Pool IDs + decode params + cache. | 3 calls |
| **S7** | Run sep-CMA-ES at the **real** `n` on a synthetic deterministic fitness (`−‖θ−θ*‖²`) for ~10 iters; assert `J` increases monotonically and λ == configured (33). | Optimizer loop, recombination, logging. | CPU |
| **S8** | Real SLM + real LLM + real reward, `m_CMA=2`, `λ=1`, `T=1`. Assert finite fitness ∈ [0,1] logged and API calls ≤ `2×5`. | End-to-end integration proof. | ~10 calls |

Two extra gates before the full run: (a) for `W=0` (uniform) θ, empirical agent/role selection is
~uniform over a few hundred samples (proves softmax factorization + sample/argmax paths); (b) confirm
fitness repeatability under greedy+cache matches expectation (surfaces the §0.3.7 caching interaction).

**Only after all gates pass:** launch per-task training (`train.py`) on GPU 5 at λ=33, m_CMA=16,
σ₀=0.1, T≈60, B_env≈31,680. Log `J(θ)` per iteration. Then `eval.py` (3 seeds) for R1–R13.

## 12. Implementation milestones (maps to tasks + JOURNAL discipline)

1. **M0 — Coordinator core (GPU5):** `slm.py` (S1), `svf.py` (S2), `head.py`, `params.py` (S3).
2. **M1 — Orchestration (no GPU):** `roles/*`, `orchestration/run.py` + `reward.py` (S4, S5).
3. **M2 — Pool integration:** finalize `llm/client.py` reasoning-effort mapping + cache (S6).
4. **M3 — Optimizer:** `optim/cmaes.py` + `fitness.py` (S7), then S8 integration.
5. **M4 — Baselines:** `optim/baselines.py` (RS, SFT, REINFORCE) for R8.
6. **M5 — Train + eval at scale** on GPU 5; collect R1–R13; write results to `experiments/`.

Every milestone appends a dated entry to `docs/JOURNAL.md` (findings, mistakes, decisions) per
`AGENTS.md` §6 — including any place reality contradicted this spec.
