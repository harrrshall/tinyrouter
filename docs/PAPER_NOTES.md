## EXTRACTION: ARCHITECTURE

I have read the full paper. I now have everything needed to write the architecture dimension report.

# TRINITY Architecture & Problem Formulation — Reverse-Engineering Notes

Source: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt` (ICLR 2026, arXiv:2512.04695v3). All numbers/equations are quoted verbatim with line citations.

---

## 1. What the coordinator decides each turn

At every turn the coordinator emits **two simultaneous discrete decisions** from a single head:

1. **Which LLM** to invoke from the pool (one of `L` models).
2. **Which role** to assign that LLM: **Thinker (T), Worker (W), or Verifier (V)**.

Verbatim (Abstract, lines 25-27): *"at each turn the coordinator assigns one of three roles (Thinker, Worker, or Verifier) to a selected LLM."*

Verbatim (Sec 3.2, lines 304-308): *"At turn k, the coordinator selects an agent (i.e., an LLM) `A_k` from the pool `M` and a role `R_k ∈ {Thinker (T), Worker (W), Verifier (V)}`. The coordinator then prepares a brief, role-specific prompt based on `C_{k-1}`, queries `A_k` to obtain a message `M_k`, and lightly post-processes `M_k` into `O_k`, which is appended to the transcript for the next turn."*

The action set is the product **agent × role**. The head produces `L + 3` logits: *"this head projects a hidden state `h ∈ R^d` to an output of size `L + 3`, which provides two sets of logits: `L` logits for selecting an LLM and three logits for assigning its role"* (Sec 3.1, lines 218-219). Note `na = 10` in the default config: 7 agents + 3 roles (lines 920-921: *"e.g., 1,024 vs. 10,240 when d_h =1024, na =10"*; line 2194: *"na = 10 logits"*).

The coordinator's **generated text is discarded** — only the head logits are used: *"our method only relies on the head's logit outputs, and the coordinator's generated text is discarded because the job of prompting is delegated to the LLMs in the pool"* (lines 235-236).

---

## 2. Formal decision process

The paper does **not** call this an MDP or bandit explicitly. It frames it as a **policy over multi-turn trajectories with a single terminal Bernoulli reward**, optimized as a black-box objective (no per-step reward, no bootstrapping). This is closer to a finite-horizon episodic policy-optimization / contextual-trajectory problem than a step-wise MDP.

### State, representation, policy (Sec 2, lines 186-204)

- **States** `s ∈ S`: *"the original query together with the full multi-turn conversation so far."*
- **Representation map**: an SLM maps each `s` to `h(s) ∈ H ⊂ R^d` *"(e.g., a penultimate-token hidden vector)."*
- **Head + policy**:
  > `f_θ : H → R^{|A|}`,  `π_θ(a | s) ∝ exp(f_θ(h(s))_a)`, `a ∈ A.`
  with parameters `θ ∈ P ⊂ R^n`, action set `A` of **agent–role pairs**.
- **Trajectory**: `τ = (s_0, a_0, ..., s_T)` with horizon `T ≤ B_turn` (fixed turn budget).
- **Reward**: terminal, **binary** — `R(τ) ∈ {0, 1}`, revealed only at end (Bernoulli).
- **Objective**:
  > `J(θ) := E_{τ∼π_θ}[R(τ)]`,  seek `θ* ∈ arg max_θ J(θ)`.

### Budget definition (lines 200-204)

> *"We regard each single, complete, end-to-end run (i.e., sampling of a trajectory τ) as an atomic evaluation, or a Bernoulli call... we seek θ* ... under a tight atomic evaluation budget `B_env` that counts individual Bernoulli calls of the terminal reward used when estimating J(θ) (e.g., via replication/averaging)."*

So: one trajectory = one Bernoulli atomic evaluation; `B_env` counts these. Each run internally makes multiple LLM calls (the cost being constrained).

### Transitions

Not given as an explicit transition kernel. Operationally: `s_{k}` is the transcript `C_{k-1} = {Q, O_1, ..., O_{k-1}}` (line 304-305); after acting, `C_k = C_{k-1} ∪ {O_k}`. The "transition" is deterministic concatenation plus a stochastic LLM generation (`A_k` produces `M_k`, post-processed to `O_k`).

---

## 3. The three roles (contracts) — Sec 3.2, lines 309-321

Each role *"enforces a distinct contract between the coordinator and the selected LLM"*:

- **Thinker strategizes** (lines 311-314): *"analyzes the current state and returns meta-level guidance, including high-level plans, decompositions, or critiques of partial solutions... it may propose a plan over subgoals, which the coordinator condenses into `O_k`... it can also specify the role of the next agent along with the plan."*
- **Worker executes** (lines 315-317): *"acts directly on the task to make concrete progress... Given `C_{k-1}`, it produces actionable content (e.g., a derivation, code snippet, or numerical result). The coordinator extracts the key information and stores it as `O_k`."*
- **Verifier evaluates** (lines 318-321): *"checks whether the accumulated solution in `C_{k-1}` is correct, complete, and responsive to Q. It outputs a judgment `u_k ∈ {ACCEPT, REVISE}` and an optional diagnosis `δ_k`. The coordinator records `(u_k, δ_k)` as `O_k` and, if `u_k = ACCEPT`, signals termination."*

---

## 4. Data flow (query → coordinator → LLM → … → answer)

From Figure 1 (lines 124-130) and Sec 3.2:

1. User query `Q` arrives.
2. Full transcript `C_{k-1} = {Q, O_1, ..., O_{k-1}}` is **concatenated** and fed to the compact coordinator SLM. Verbatim (lines 137-138): *"T RINITY selects an LLM and constructs its input by concatenating the original query with the full transcript of prior turns."*
3. SLM produces a hidden state `h` at the **penultimate output token** position; the lightweight head reads `h` → `L+3` logits.
4. Head selects agent `A_k` and role `R_k`.
5. A **message-processing module injects a role-specific prompt** before sending to the chosen LLM (Fig 1 caption, lines 126-128: *"A message processing module injects a role-specific prompt before the request is sent to the chosen LLM"*).
6. `A_k` returns message `M_k`; coordinator lightly post-processes into `O_k`; appended to transcript.
7. Loop to next turn until termination.

---

## 5. Pool size & role assignment mechanism

- **Pool size in experiments: 7 models** (`L = 7`). Verbatim (lines 383-396): *"Our model pool contains seven models... three top-tier closed-source models (GPT-5, Gemini-2.5-pro, Claude-4-Sonnet)... and four well-known open-source models (Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B, Qwen-3-32B (direct), and Qwen-3-32B (reasoning))."*
  - Note: "Qwen-3-32B (direct)" and "Qwen-3-32B (reasoning)" are counted as **two distinct pool entries** (same weights, different inference mode). Agent IDs (Fig 15, lines 2435-2437): A0=GPT-5, A1=Claude-Sonnet-4-20250514, A2=Gemini-2.5-pro, A3=DeepSeek-R1-Distill-Qwen-32B, A4=Gemma-3-27b-It, A5=Qwen3-32B (reasoning), A6=Qwen/Qwen3-32B (direct).
- **Role assignment**: the head's three role logits are converted to a choice via softmax (default) or argmax. There is **no explicit per-LLM role restriction** — any of the 7 agents can be assigned any of the 3 roles each turn. The role is determined purely by the head logits given `h(s)`.
- **"One of three roles"** = the 3 role logits; the generic action is a (agent, role) pair drawn from `π_θ`.

> **Replication-context note (not from paper):** Our setup uses **3** Fireworks models (`deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6`), so `L = 3` and the head outputs `L+3 = 6` logits (`na = 6`). The paper's default `na = 10` assumes `L = 7`. This is a deliberate scaling-down, not a paper value.

---

## 6. Termination / session end

Two stopping conditions (Sec 3.2, lines 330-331; Intro lines 141-143):

> `τ = min{ k ≤ K : R_k = V and u_k = ACCEPT }`, with `τ = K` if no acceptance occurs.

Verbatim (lines 141-143): *"The process halts when the verifier is selected and accepts current response as the final answer, or when a fixed-turn budget is exhausted."* The final answer returned to the user is `O_τ` (line 331).

- **Max turns `K = 5`** in main experiments. Verbatim (lines 403-404): *"We also set the maximum number of coordination turns to five."* Also lines 2378: random baseline *"max turn number is 5, same as T RINITY setting."*
- Turn-budget ablation (Fig 4 bottom, lines 825-826): performance improves *"from 0.823 to 0.863 as turns increase from 2 to 6"* — so `K ∈ {2, 4, 6}` were swept.

---

## 7. The coordinator loop (pseudocode)

Reconstructed from Sec 2 + Sec 3.2. All control values cited above.

```
# Fixed config (paper main experiments):
#   K = 5                         # max coordination turns (line 404)
#   pool M, |M| = L = 7           # (lines 383-396); replication: L = 3
#   roles = {T, W, V}             # (line 306)
#   max_gen_tokens = 4096/LLM     # (lines 402-403), "minimal reasoning effort"
#   head outputs L + 3 logits     # (lines 218-219); na = 10 when L = 7

def trinity_run(Q, theta):                      # one trajectory = one Bernoulli atomic eval
    transcript C = [Q]                          # s_0  (lines 304-305)
    O_final = None
    for k in 1..K:
        s = concat(C)                           # full transcript (lines 137-138)
        h = SLM.forward(s).hidden[penultimate_output_token]   # h(s) in R^d (line 187, Fig 2)
        agent_logits, role_logits = head_theta(h)             # L + 3 logits (line 219)
        A_k = sample_or_argmax(softmax(agent_logits))         # pi_theta over agents
        R_k = sample_or_argmax(softmax(role_logits))          # T / W / V  (default softmax; argmax for block-diag-10, line 928)
        prompt_k = inject_role_prompt(R_k, C)                 # message-processing module (Fig 1 caption)
        M_k = LLM[A_k].generate(prompt_k, max_tokens=4096)    # query selected LLM (line 308)
        O_k = postprocess(M_k, R_k)                           # condense/extract/record (lines 313-320)
        C.append(O_k)
        O_final = O_k
        if R_k == V and verifier_judgment(O_k) == ACCEPT:     # termination rule (line 330)
            break
    return O_final                              # returned to user (line 331)

# Reward (revealed only at trajectory end):
def reward(O_final, ground_truth):
    return 1 if correct(O_final, ground_truth) else 0         # R(tau) in {0,1} (line 196)

# Optimization (outer loop, sep-CMA-ES) — see optimization dimension:
#   J(theta) = E_tau[R(tau)]   maximized under atomic budget B_env (lines 197, 203)
```

---

## 8. Key dimensions / counts relevant to architecture

| Quantity | Value | Citation |
|---|---|---|
| Coordinator SLM | Qwen3-0.6B (≈0.6B params) | lines 23, 134, 379 |
| Total learnable params | **< 20K** ("under 20K") | lines 24, 136, 176, 225 |
| Head params (linear, default) | **10K** ("≈10K"); exactly `d_h·na` = 10,240 for `d_h=1024, na=10` | lines 24, 354, 380, 920-921, Table 6 line 2209 |
| Hidden dim `d_h` | 1024 (Qwen3-0.6B) | lines 921, 2092 |
| SVF (singular-value fine-tuning) params | 9,216 | Table 6, line 2209 |
| SVF target layer | "second-to-last layer of the 0.6B model" | line 381 |
| Head input token | **penultimate output token** hidden state | lines 59-61, 273, 290-291, Fig 2 |
| `na` (head output logits) | `L + 3` = 10 (with `L=7`) | lines 219, 921 |
| Optimization dim `n` | ≈ 10000 | lines 152, 359, 1233, 1436 |
| Max turns `K` | 5 | line 404 |
| Max gen tokens / LLM | 4096, "minimal reasoning effort" | lines 402-403 |
| Reward | binary terminal `R(τ) ∈ {0,1}` | line 196 |

**Head architectures available** (Appendix A.4, lines 2119-2218; param sizes Table 6, line 2209): linear (10,240) [default, eq. 5 `z=Wh`, `W∈R^{na×d_h}`, no bias], low-rank (20,680; `r=14`, ELU `α=0.1`, eq. 6-8), sparse (11,266; Gumbel top-k, `τ∈[1.0,20.0]`, eq. 9-11), block-diagonal-2 (5,120), block-diagonal-10 (1,024; one block per agent/role, argmax conversion). Default head = **linear** (line 380, best overall per Table 3).

---

## 9. OPEN / AMBIGUOUS (needed to implement, absent or under-specified)

1. **Exact role-specific prompt templates.** The paper repeatedly references "a brief, role-specific prompt" injected by a "message processing module" (lines 126-128, 306-308) and shows one *example* of generated text in Fig 1 (lines 82-120), but **no verbatim Thinker/Worker/Verifier prompt template is given anywhere** in the text or appendix. The example coordinator message "The user requires the calculation... The next agent should act as a solver..." (lines 82-84) is illustrative output, not a template. → Must be reconstructed; no quotable template exists.

2. **Post-processing ("lightly post-processes `M_k` into `O_k`") specifics.** Described only as "condenses," "extracts the key information," "records `(u_k, δ_k)`" (lines 313-320). The actual truncation/summarization procedure is unspecified.

3. **Sampling vs argmax at inference for the head.** Default output conversion is softmax (line 928: *"By default, the output conversion is softmax normalization"*), and block-diagonal-10 uses argmax. But whether action selection during a deployed run is **argmax over softmax** or **stochastic sampling** from `π_θ` is not stated explicitly for the linear head. The MDP definition uses `π_θ(a|s)` (stochastic), but inference behavior is ambiguous.

4. **Verifier ACCEPT/REVISE extraction.** `u_k ∈ {ACCEPT, REVISE}` (line 319) is produced by the Verifier LLM, but **how the textual verifier output is parsed into the ACCEPT/REVISE signal** (keyword match? structured output? a separate classifier?) is not described.

5. **Whether agent/role are decided jointly or the role logits are conditioned on the agent.** The head emits both sets of logits in parallel from the same `h` (lines 218-219); it appears independent (two separate softmaxes), but the paper never explicitly states the joint vs. factored sampling. Block-diagonal-10 (one block per agent/role logit) implies independence is *desirable*, but the default linear head couples all logits through one `W`.

6. **Initial turn behavior.** Fig 1 shows the first coordinator message noting "The initial response is empty" (line 83); whether turn 1 is forced to a particular role (e.g., Thinker) or freely chosen by the head is not stated. The example happens to use T→W→V but this is not declared as an enforced ordering.

7. **`B_turn` vs `K`.** Sec 2 uses `B_turn` (turn budget, line 195); Sec 3.2 uses `K` (line 304). They appear to be the same quantity (`K = B_turn = 5`) but this identity is implied, not stated.

8. **Reward correctness function `R(τ)`.** Binary `{0,1}` (line 196), but the per-task verification (exact-match, pass@1 test execution, judge model for MT-Bench, etc.) is implicit in the benchmark choice and never centralized into one definition.

9. **Replication-context mismatches (ours, flagged for the team):** Paper uses `L=7` (→ `na=10`, head ≈10,240 params, `n≈10000`). Our 3-model Fireworks pool gives `L=3` → `na=6` → head = `6·1024 = 6,144` params, and total optimization dimension `n` shrinks accordingly. The paper's sep-CMA-ES `λ=⌈4+3 ln n⌉=32`, `m_CMA=16`, `m_RS=32` were derived for `n≈10000` (lines 359, 1436) — these **will not transfer unchanged** to the smaller dimension. (Optimization dimension owns this; flagged here because it changes head output size.)

---

## EXTRACTION: COORDINATOR

I now have the complete paper. Here are my findings on the COORDINATOR dimension, with everything quoted verbatim and cited.

---

# TRINITY Coordinator — Reverse-Engineering Findings

All quotes cite section/page of `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt` (line numbers given in brackets).

## 1. The compact ~0.6B SLM — exact model

> "We use **Qwen3-0.6B** (Yang et al., 2025) as the coordinator's SLM" — §4.1 Experimental Setup [L379].

Confirmed again in Table 6 caption: "The size is calculated based on the SLM **Qwen3-0.6B**" [L2205-2206]. The reference is "Qwen3 technical report. arXiv:2505.09388, 2025" [L1194-1196]. No further version tag (e.g. instruct vs base) is given.

**Note for our replication:** This is a LOCAL model on the H200, NOT a Fireworks model. The three Fireworks models (deepseek-v4-pro, glm-5p2, kimi-k2p6) are the *coordinated* LLM pool — but note the paper's actual pool is 7 *different* models (GPT-5, Gemini-2.5-pro, Claude-4-Sonnet, Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B direct, Qwen3-32B reasoning) [L383-395]. Our pool size and identities differ from the paper.

## 2. Hidden-state extraction (which layer, which token, pooling, pre/post-generation)

**Token position:**
> "It takes the hidden state h corresponding to the **penultimate output token** as its sole input." — Figure 2 caption [L290-291].

> "For output sequences, the **penultimate token's hidden state** carries rich context. It attends over the entire sequence and guides the prediction of a special token (such as `<\think>` or the EOS token), ensuring a stable output distribution." — §1 Introduction [L59-61].

**Which layer:** The head is appended after the final hidden layer:
> "we append a lightweight head directly **after the coordinator SLM's final hidden layer**." — §3.1 [L217].

So `h ∈ R^d` is the final-layer hidden state at the penultimate output-token position. `d = 1024` (Qwen3-0.6B hidden size), confirmed by the separability experiment: "the coordinator's representation space (**1024 dimensions**, 7 agent classes, 4 task type clusters)" [L2092] and the head equations use `d_h = 1024` [L920, L2213-2214].

**Pooling:** None. The head takes a single token's hidden vector ("**its sole input**" [L291]; "the hidden state at the position marked by '<Head Input>' is the input to lightweight head" [L295-296]). No pooling, no averaging.

**Pre- vs post-generation — IMPORTANT subtlety:**
- The separability analysis (Figure 5) uses **input-side** penultimate-token states: "Both are based on **penultimate-token hidden states processed by the SLM on the input sequence**" [L893-894].
- But the design explicitly allows an *earlier* token (not waiting for full generation), because the SLM's generated text is discarded:
> "our method only relies on the head's logit outputs, and the coordinator's generated text is discarded… Rather than waiting for a full generation, this allows the coordinator to take hidden states corresponding to an **earlier token instead of the penultimate** to make a quick decision." — §3.1 [L235-238].

So the design is: feed the full transcript (query + prior turns) as the SLM input, take the penultimate-token hidden state from the final layer. The "output token" framing means the SLM does a forward pass over `[input ... <Head Input> <EOS>]` (Figure 2 shows input `<BOS>...<Head Input><EOS>` [L286-287]); the penultimate-token state is the `<Head Input>` position. The last-token ablation (using the EOS position instead) collapses performance:

> "(4) use the last instead of the penultimate token… switching to the final token, which often corresponds to a semantically sparse EOS token, causes a severe performance collapse, particularly on LiveCodeBench (more than 10 points drop)." — §4.5, Table 2 [L811, L836-837]. (w/ Last token row: LiveCodeBench 50.85 vs Trinity 61.46 [L818, L814].)

## 3. The lightweight head — exact architecture

**Default / best head = `linear`** (single linear layer, no bias):

> "paired with a **single linear layer of 10K parameters** as the simple but effective head" — §4.1 [L380].

Equation 5 [L2124-2125]:
> z = Wh,  W ∈ R^(n_a × d_h)

> "This head has exactly **d_h · n_a** trainable parameters… It allows unrestricted linear combinations of hidden dimensions to express agent and role preferences" [L2127-2128]. No bias, no activation.

With `d_h = 1024`, `n_a = 10` → **10,240 parameters** (Table 6: `linear = 10240` [L2208-2209]).

**Output dimension n_a = L + 3:**
> "this head projects a hidden state h ∈ R^d to an output of size **L + 3**, which provides two sets of logits: **L logits for selecting an LLM and three logits for assigning its role**." — §3.1 [L218-219].

For the paper's L = 7 agents: n_a = 7 + 3 = 10. Confirmed: "our setting with **na = 10 logits**" [L2194], "e.g., 1,024 vs. 10,240 when d_h =1024, **na =10**" [L920].

**For OUR replication:** L = 3 (deepseek-v4-pro, glm-5p2, kimi-k2p6) → n_a = 3 + 3 = **6**, so linear head W ∈ R^(6×1024) = **6,144 parameters**.

**Output → decision mapping:**
> "producing two sets of logits, one to select an LLM from the pool of L models, and another to assign one of three roles." — Figure 2 caption [L292-293].

- LLM selection: first L logits → softmax (default) → pick agent. Figure 2 shows "LLM logits → R^L" and "Role logits → R^3" [L262-263, L270-271].
- Role selection: last 3 logits → softmax → pick role ∈ {Thinker, Worker, Verifier}.
- Default output conversion is **softmax**: "By default, the output conversion is softmax normalization." — Table 3 caption [L927-928]. The policy is `π_θ(a|s) ∝ exp(f_θ(h(s))_a)` — §2 [L192].

**Stopping decision:** There is NO separate "stop" head output. Termination is implicit in the role+verifier mechanism, NOT a learned logit:
> "The termination time is τ = min{ k ≤ K : R_k = V and u_k = ACCEPT }, with τ = K if no acceptance occurs." — §3.2 [L330].

The ACCEPT/REVISE judgment `u_k` comes from the **Verifier LLM's text output**, not the head:
> "It outputs a judgment u_k ∈ {ACCEPT, REVISE}… The coordinator records (u_k, δ_k) as O_k and, if u_k = ACCEPT, signals termination." — §3.2 [L319-321].

So the head outputs ONLY (agent, role). Stopping happens when the head picks Verifier AND the verifier LLM says ACCEPT, OR the turn budget K=5 is hit.

**Head inputs:** ONLY the hidden-state vector. "its **sole input**" [L291]. No turn index, no history features, no model/role one-hots are fed to the head. (The transcript context is already encoded in `h` because the full transcript is the SLM input — see §4 below.)

## 4. The ~10K trainable parameters CMA-ES optimizes

There are **two** trainable parameter sets (§3.1 [L216-224]); total "**under 20K**" [L24, L136, L177, L225]:

**(a) The head** — 10,240 params (linear) [L2209].

**(b) Singular Value Fine-tuning (SVF) of the SLM** — 9,216 params [L2208-2209]:
> "we adapt a small set of the backbone's layers using a **singular value fine-tuning** approach. For a selected subset of the coordinator SLM's weight matrices, we perform a **singular value decomposition and only learn the singular value scales, keeping the orthogonal matrices fixed**." — §3.1 [L221-224]. (Method from "Transformer-squared / Sun et al. 2025" [L1167-1169].)

> "select the **second-to-last layer** of the 0.6B model for singular value fine-tuning." — §4.1 [L380-381].

So CMA-ES jointly optimizes a vector θ ≈ 10,240 + 9,216 = **19,456 params** (Table 6: SVF=9216, linear=10240 [L2208-2209]). The theory analysis repeatedly calls it "a coordination head with about **10K parameters**" and "n ≈ 10000" [L352, L359, L1233, L1436] — this "10K" refers primarily to the head; the full optimized vector is ~19.5K. The "weak coupling among parameters" / block-ε-separability is the key property [L144-145, L341].

SVF ablation: removing it drops average 70.44 → 67.85 [L814-815].

## 5. Alternative head architectures (Appendix A.4, for ablation completeness)

All map `h ∈ R^d_h` → logits `z ∈ R^n_a` [L2121-2122]. Param counts in Table 6 [L2208-2209]:

| Head | Equation | Params (d_h=1024, n_a=10) |
|---|---|---|
| **linear** (default) | z = Wh, W∈R^(n_a×d_h), no bias | **10,240** |
| low-rank | u=ELU(Uh,α=0.1); z=Vu·σ; U∈R^(r×d_h), V∈R^(n_a×r), **r=14**, σ fixed non-trainable scale; Xavier-uniform init | 20,680 |
| sparse | z=W(h⊙α), α∈R^d_h learnable; k=max(1,⌊d_h·(1−sigmoid(ρ))⌋); Gumbel top-k, temp τ∈[1.0,20.0]; hard top-k at inference | d_h·n_a+d_h+2 = 11,266 |
| block-diagonal-2 | B=2 disjoint blocks, proportional partition | 5,120 |
| block-diagonal-10 | B=10 (one block/logit), a_i=1, + **argmax** output | 1,024 |

Quotes: low-rank "we fix the bottleneck to **r = 14**" [L2138], ELU "α = 0.1" [L2133]; sparse "d_h n_a + d_h + 2 parameters" [L2169-2170]; block-diag-10 "It creates one block per agent/role (B = 10, a_i = 1)" [L2195]. Head performance (Table 3): linear is best overall — LiveCodeBench 0.615, MATH500 0.880, MMLU 0.916, RLPR 0.401 [L930]; block-diagonal-10+argmax: 0.551/0.812/0.802/0.376 [L934].

## 6. CMA-ES configuration (the optimizer over the ~10–20K params)

> "separable CMA-ES (**sep-CMA-ES**)" with **diagonal covariance** [L148-149, L349].
- Population size: **λ = ⌈4 + 3 ln n⌉** = ⌈4 + 3 ln 10000⌉ = **32** [L354, L1246, L1436].
- Replication count (evals per candidate): **m_CMA = 16** [L359, L1233, L1437].
- Iterations: **60** ("ran for 60 iterations" for the budget-matched comparison [L2231]; theory uses T∈[2,60] [L1452]).
- Diagonal learning rate: **c_cov = Θ(1/n)** [L1413, L1484].
- Budget regime: "**1.5k–40k evaluations for a 10k-dimensional problem**" [L152].
- RS baseline (for comparison) used m_RS=32, sampling uniformly from [−0.5,0.5] [L359, L2234-2237].

## 7. Multi-turn protocol & role contracts (context that defines the head's action space)

- Max turns **K = 5**: "we also set the maximum number of coordination turns to **five**" [L403-404].
- At turn k, head selects agent A_k from pool M and role R_k ∈ {Thinker(T), Worker(W), Verifier(V)} [L305-306]. SLM input = full transcript C_{k-1} = (Q, O_1,…,O_{k-1}) [L137, L304-308].
- Role contracts [L311-321]: Thinker → meta-level plans/decompositions/critiques; Worker → concrete actionable content; Verifier → u_k∈{ACCEPT,REVISE}+optional diagnosis δ_k.
- A "message processing module injects a role-specific prompt" then queries the chosen LLM; output M_k is "lightly post-processed" into O_k [L127-128, L307-308].
- Default max generated tokens per LLM = **4096**, "with **minimal reasoning effort**" [L402-403].

---

# OPEN / AMBIGUOUS (details absent or under-specified for faithful re-implementation)

1. **Exact role-specific prompt templates.** The paper references a "message processing module" that "injects a role-specific prompt" and "lightly post-processes" outputs [L127-128, L307-308, L313-317], and Figure 1 shows *example* paraphrased outputs [L82-120], but **the verbatim prompt strings for Thinker/Worker/Verifier and the post-processing rules are never given.** This is the single biggest gap for the coordinator dimension. Must be reconstructed.

2. **How ACCEPT/REVISE is parsed from Verifier text.** Stated as u_k∈{ACCEPT,REVISE} [L319] but the exact parsing rule / required output format from the verifier LLM is not specified.

3. **Penultimate-token mechanics in practice.** It is stated the head can use "an earlier token instead of the penultimate to make a quick decision" [L237-238] for speed, while separability analysis uses input-side penultimate-token states [L893-894]. **Which token is used during actual coordination training/inference (penultimate of input, or a forced short generation) is not pinned down** — the design permits both. Figure 2 input is `<BOS>...<Head Input><EOS>` implying a constructed sequence, but how `<Head Input>`/`<EOS>` are appended to the raw transcript is unspecified.

4. **SVF specifics.** "second-to-last layer" is selected [L380-381], yielding 9,216 trainable singular-value scales [L2209], but **which weight matrices within that layer** (Q/K/V/O/MLP?) get SVD'd is not enumerated. 9,216 = e.g. 9×1024 or 6×1536 — the decomposition is not detailed beyond "a selected subset of weight matrices" [L222].

5. **Softmax temperature for agent/role selection.** Policy is exp(logits) [L192], default conversion softmax [L927], but no temperature value, and whether selection is argmax vs sampled at inference is not stated (block-diagonal-10 uses argmax explicitly [L928], implying linear may use sampling/argmax — unclear).

6. **CMA-ES init (mean θ_0, initial σ).** Initial step-size σ_0 and the head/SVF initialization for the *trained* run are not given (Xavier init mentioned only for the low-rank head's ablation [L2148-2149]; RS init range [−0.5,0.5] is for the RS baseline, not CMA-ES [L2235]). Total budget B_env exact value within "1.5k–40k" [L152] not pinned for each task.

7. **Weight recombination / parent count.** sep-CMA-ES uses "µ parents with weights (w_j)" [L1246] but µ and the weight scheme are left to the default sep-CMA-ES (Ros & Hansen 2008); not restated numerically.

8. **Our-pool mismatch.** Paper uses L=7 specific models (none of which are deepseek-v4-pro/glm-5p2/kimi-k2p6). With L=3 our head is R^(6×1024)=6,144 params and n_a=6 — we are deviating from the paper's exact dimensions by design of the replication context. No paper number covers a 3-model pool.

---

## EXTRACTION: ROLES

I have now read the complete paper. I have all the content relevant to the ROLES dimension. Note that the paper does NOT contain verbatim role prompt templates anywhere (I checked the appendix exhaustively, A.1 through A.7.4). Let me compile the findings.

# TRINITY — ROLES & Multi-Turn Protocol (faithful re-implementation reference)

Source: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt` (ICLR 2026, Sakana AI). Section/line citations below refer to that file. **Every number/equation is quoted verbatim; nothing is invented.** Items the paper leaves unspecified are collected under OPEN / AMBIGUOUS at the end.

---

## 1. The three roles — verbatim definitions

The roles are introduced in the abstract and Section 3.2 (Tri-role coordination, lines 298–321).

> "at each turn the coordinator assigns one of three roles (Thinker, Worker, or Verifier) to a selected LLM, effectively offloading complex skill acquisition from the coordinator itself." (Abstract, lines 26–27)

The formal contract (Section 3.2, lines 309–321), quoted verbatim:

> "In TRINITY, we define three roles, namely Thinker, Worker, and Verifier, each of which enforces a distinct contract between the coordinator and the selected LLM:
> - **Thinker strategizes.** The thinker analyzes the current state and returns meta-level guidance, including high-level plans, decompositions, or critiques of partial solutions. Formally, it may propose a plan over subgoals, which the coordinator condenses into Ok to steer subsequent turns, it can also specify the role of the next agent along with the plan.
> - **Worker executes.** The worker acts directly on the task to make concrete progress toward a final solution. Given Ck−1, it produces actionable content (e.g., a derivation, code snippet, or numerical result). The coordinator extracts the key information and stores it as Ok.
> - **Verifier evaluates.** The verifier checks whether the accumulated solution in Ck−1 is correct, complete, and responsive to Q. It outputs a judgment uk ∈ {ACCEPT, REVISE} and an optional diagnosis δk. The coordinator records (uk, δk) as Ok and, if uk = ACCEPT, signals termination."

Role abbreviations used throughout: **T = Thinker, W = Worker, V = Verifier** (Figure 1 caption, lines 126–127).

Figure 1 worked example (lines 76–123) shows the concrete behavior on a depreciation problem:
- **Turn 1 — Thinker (T):** produces a step-by-step plan ("Here is a step-by-step plan to solve the problem: 1. Determine the straight-line depreciation rate … 2. Calculate the double declining-balance rate …").
- **Turn 2 — Worker (W):** performs the calculation ("1. Calculate the straight-line depreciation rate … 2. Calculate the double declining-balance rate … This is twice the straight-line rate …").
- **Turn 3 — Verifier (V):** validates and raises an edge case ("The calculation provided in the current response appears to be correct … However, one potential issue to verify is whether the double declining-balance method should switch to straight-line depreciation if it results in a higher depreciation amount in later years …").

The first message-processing box (lines 82–84) shows the coordinator's internal framing seeded at Turn 1: *"The user requires the calculation… The initial response is empty. The next agent should act as a solver to perform the calculation."* — note this references the worker as a "solver." This is the only near-prompt text in the paper.

---

## 2. Turn-by-turn protocol

### 2.1 Notation and state passing (Section 3.2, lines 304–308)

> "Coordination proceeds over at most K turns for a given user query Q. Let the transcript after k−1 turns be Ck−1 = (Q, O1, …, Ok−1). At turn k, the coordinator selects an agent (i.e., an LLM) Ak from the pool M and a role Rk ∈ {Thinker (T), Worker (W), Verifier (V)}. The coordinator then prepares a brief, role-specific prompt based on Ck−1, queries Ak to obtain a message Mk, and lightly post-processes Mk into Ok, which is appended to the transcript for the next turn."

So the per-turn loop is:
1. Build context `C_{k-1} = (Q, O_1, …, O_{k-1})` — the original query **concatenated** with the full transcript of all prior turn outputs. (Also Intro, lines 136–138: *"At each turn, TRINITY selects an LLM and constructs its input by concatenating the original query with the full transcript of prior turns."*)
2. SLM ingests the context; the lightweight head emits logits → selects agent `A_k` (from L/`|M|` logits) and role `R_k` (from 3 logits). (Section 3.1, lines 217–219; Figure 2 caption lines 289–293.)
3. Coordinator constructs a **brief, role-specific prompt** based on `C_{k-1}` (the role-prompt construction; the actual template text is not given — see OPEN).
4. Query `A_k` → message `M_k`.
5. Light post-processing of `M_k` → `O_k`, appended to transcript.

### 2.2 Decision input (which token's hidden state drives the head)

> "It takes the hidden state h corresponding to the penultimate output token as its sole input." (Figure 2 caption, lines 290–291)

> "For output sequences, the penultimate token's hidden state carries rich context. It attends over the entire sequence and guides the prediction of a special token (such as `<\think>` or the EOS token)…" (Section 1, lines 59–61)

The coordinator's *own* generated text is discarded; only the head logits are used, and an earlier-than-penultimate token may be used for speed:

> "our method only relies on the head's logit outputs, and the coordinator's generated text is discarded because the job of prompting is delegated to the LLMs in the pool … this allows the coordinator to take hidden states corresponding to an earlier token instead of the penultimate to make a quick decision." (Section 3.1, lines 235–238)

Ablation confirms the penultimate token is required: using the **last token** (often the EOS) causes severe collapse, especially LiveCodeBench (−10 points). (Table 2 row "w/ Last token": LCB 50.85 vs 61.46; Section 4.5 lines 836–837.)

### 2.3 Termination condition (Section 3.2, lines 330–331)

> "The termination time is τ = min{ k ≤ K : Rk = V and uk = ACCEPT }, with τ = K if no acceptance occurs. The final answer returned to the user is Oτ."

So the process halts when (a) a **Verifier is selected and returns `u_k = ACCEPT`**, or (b) the **turn budget K is exhausted**. The final answer is the output `O_τ` of the terminating turn. (Restated in Intro lines 141–143: *"The process halts when the verifier is selected and accepts current response as the final answer, or when a fixed-turn budget is exhausted."*)

### 2.4 Turn budget / horizon

- Formal: trajectory `τ = (s0, a0, …, sT)` with `T ≤ B_turn`, `B_turn` = fixed turn budget. (Section 2, lines 194–195.)
- **Default max turns = 5** (training/eval): *"We also set the maximum number of coordination turns to five."* (Section 4.1, lines 403–404). Random-agent baseline also uses max turn = 5 (A.7.1, line 2378).
- LiveCodeBench "unleashing full power" sweep (Figure 4 bottom, lines 825–826): max turns swept over **2, 4, 6**; performance improves **0.823 → 0.863** as turns go 2 → 6. Stated range *"improving from 0.823 to 0.863 as turns increase from 2 to 6."*

---

## 3. Action space and head outputs (relevant to role selection)

- Action set `A` = agent–role pairs. Policy: `πθ(a | s) ∝ exp(fθ(h(s))_a)`, `a ∈ A`. (Section 2, lines 189–192.)
- Head maps `h ∈ R^d` → output of size **L + 3**: *"this head projects a hidden state h ∈ Rd to an output of size L + 3, which provides two sets of logits: L logits for selecting an LLM and three logits for assigning its role."* (Section 3.1, lines 217–219.)
- With the experimental pool L = 7, `n_a = L + 3 = 10` logits (confirmed: *"na = 10 logits"*, block-diagonal-10 description, line 2194; and *"na =10"* line 920).
- Default output conversion is **softmax** over the logits; block-diagonal-10 uses **argmax** instead. (Table 3 caption, lines 927–928.) The argmax variant decouples blocks: *"With argmax, decisions depend only on the largest logit…"* (Section 4.7, lines 921–922).

**OPEN nuance:** With softmax, whether the L agent-logits and the 3 role-logits are softmaxed **separately** (two independent categoricals) vs. jointly is not stated explicitly; the wording "two sets of logits" plus the policy `πθ(a|s)` over agent–role pairs `A` suggests two separate categorical heads (one over L agents, one over 3 roles) sampled/argmaxed independently per turn, but the exact factorization and whether selection is sampled or argmaxed at inference is not pinned down. See OPEN.

---

## 4. Role / agent selection behavior (learned, empirical — for sanity-checking a reimplementation)

These are *observed* distributions, not config, but useful to validate that a reimplementation's learned roles look right.

### Per-task agent selection (Figure 15, A.7.2, lines 2414–2437). Values are Selection %.
Agent legend: A0 = GPT-5, A1 = Claude-Sonnet-4-20250514, A2 = Gemini-2.5-pro, A3 = DeepSeek-R1-Distill-Qwen-32B, A4 = Gemma-3-27b-It, A5 = Qwen3-32B (reasoning), A6 = Qwen3-32B (direct).

| Task | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| LCB | 87.9 | 9.3 | 2.3 | 0.0 | 0.0 | 0.0 | 0.5 |
| math500 | 9.2 | 5.3 | 82.2 | 0.0 | 0.0 | 0.0 | 3.4 |
| mmlu | 24.7 | 74.3 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rlpr | 22.2 | 16.3 | 59.4 | 0.0 | 0.0 | 0.0 | 2.1 |

Note: on LiveCodeBench the learned selection narrows to the three closed models (Section 4.4, lines 794–795), enabling the unconstrained 86.2% run without retraining.

### Role ablations (Table 2, lines 813–818) — establishing each role's contribution:
| Method | LiveCodeBench | MATH500 | MMLU | RLPR | Average |
|---|---|---|---|---|---|
| TRINITY (full tri-role) | 61.46 | 88.00 | 91.56 | 40.72 | 70.44 |
| w/o Thinker-role selection | 57.80 | 86.20 | 92.75 | 38.00 | 68.69 |
| w/o Tri-role selection | 58.28 | 82.00 | 91.64 | 36.15 | 67.02 |
| w/ Last token (vs penultimate) | 50.85 | 87.00 | 82.19 | 38.60 | 64.66 |
| w/o agent selection (role selection only, single fixed agent) | (row 5 in caption) | | | | |

Removing roles is detrimental to complex reasoning: *"causing substantial degradation on MATH500 (-6.0 points) and RLPR (-4.57 points)"* (Section 4.5, lines 834–835). The caption (lines 809–812) lists 5 ablations including "(5) remove agent selection but keep role selection," which "significantly undermined" performance (lines 838–839), but the numeric row for ablation (5) is **not printed** in Table 2 as shown.

---

## 5. Decoding params / token budgets per role

The paper gives **global** generation settings applied to every LLM in the pool, but **no per-role decoding parameters**:

- *"we set the default maximum generated tokens to **4096** for each LLM, with **minimal reasoning effort**."* (Section 4.1, lines 402–403).
- Baseline "Individual Agent" evaluated at **4K and 20K (5×)** max-token limits (A.7.1, lines 2372–2374). The 5× = 20K corresponds to max-turn 5. (Section 4.3, line 733: *"The 5× inference budget matches our maximum turn setting of five."*)
- LiveCodeBench "full power": the 4096 constraint is **removed** (no retraining). (Section 4.4, lines 796–797.)

Observed average output tokens for TRINITY coordination runs (Table 9, lines 2480–2486): Math500 = 2,853; MMLU = 1,200; RLPR = 2,141; LiveCodeBench = 1,999. (These are end-to-end per-question totals across turns, not per-role.)

**No temperature, top-p, top-k, or per-role max-token value appears anywhere in the paper.** See OPEN.

---

## 6. Coordinator config that bears on role decisions (for completeness)

- Coordinator SLM = **Qwen3-0.6B** (Section 4.1, line 379), second-to-last layer used for singular value fine-tuning (line 381).
- Head = single **linear** layer, **10K params** ("10,240" for d_h=1024, n_a=10; Table 6, line 2209). Hidden dim `d_h = 1024` (Sections 4.7 / A.4).
- Total learnable params **< 20K** (Abstract line 24; Intro line 136). SVF params = 9,216 (Table 6).
- Optimizer = **sep-CMA-ES**, population `λ = ⌈4 + 3 ln n⌉ = 32` (n≈10000), replication `m_CMA = 16` (Section 3.3 lines 354–362; A.1.2 line 1436).

---

## OPEN / AMBIGUOUS (details needed to implement roles/protocol that the paper does NOT specify)

1. **Verbatim role prompt templates are absent.** The paper repeatedly says the coordinator "prepares a brief, role-specific prompt" (lines 306–308) and Figure 2 caption defers to "see Appendix A.4" — but A.4 is *Head Architecture Design* (math only, lines 2119–2218), **not** prompt templates. There is **no system prompt or role-prompt template anywhere** in the appendix (A.1–A.7 fully read). The only quasi-prompt text is the Figure 1 message-processing snippet ("The next agent should act as a solver…", lines 82–84). The exact Thinker/Worker/Verifier system prompts must be reconstructed/invented — not available verbatim.

2. **"Light post-processing" of M_k → O_k is undefined.** The paper says the coordinator "lightly post-processes Mk into Ok" / "condenses into Ok" / "extracts the key information" (lines 307–317) but never specifies the algorithm. Whether this is truncation, LLM-based summarization, regex extraction, or pass-through is unspecified.

3. **Verifier ACCEPT/REVISE extraction is undefined.** `u_k ∈ {ACCEPT, REVISE}` and diagnosis `δ_k` are stated (lines 318–321), but **how** the judgment is parsed from the Verifier LLM's free-text output (keyword match? structured output? a classifier?) is not described. The example Verifier output (lines 110–120) is prose with no explicit ACCEPT/REVISE token visible.

4. **Sampling vs argmax at inference, and softmax factorization.** Whether agent/role are sampled from `πθ` or taken as argmax at test time is not stated for the default (linear/softmax) head. Whether agent-logits and role-logits are normalized as two separate categoricals is implied but not explicit (§3 above).

5. **Per-role decoding parameters.** No temperature / top-p / top-k / per-role max_tokens given. Only a global `max_tokens=4096` and "minimal reasoning effort" (lines 402–403). The mapping of "minimal reasoning effort" to API params (e.g., reasoning_effort, thinking budget) for each provider is unspecified.

6. **First-turn behavior / forced roles.** The example seeds Turn 1 with a Thinker, but the paper does not state whether the first turn is constrained to a particular role or freely chosen by the head. Likewise, whether a Verifier-ACCEPT can fire on Turn 1 (empty/near-empty solution) is unspecified — the formal rule `τ = min{k ≤ K : R_k=V and u_k=ACCEPT}` allows it but no guard is described.

7. **What is returned if τ = K with no ACCEPT.** Rule says final answer = `O_τ = O_K` (last turn's output) regardless of role of turn K (lines 330–331). It does not say whether a final answer-extraction/formatting pass is run, nor how a non-Verifier, non-Worker final turn's `O_K` is converted to the user-facing answer.

8. **Mapping to the replication context.** The paper's pool is GPT-5 / Gemini-2.5-pro / Claude-4-Sonnet + 4 OSS models. The replication uses Fireworks `deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6` (3 models) — the paper's L=7 / n_a=10 head sizing and the learned per-task agent distributions (Fig 15) will not transfer; head output must be re-sized to L=3 → n_a=6, and the agent-selection priors retrained. This is an implementation mismatch, not a paper ambiguity, but flagged because the "ROLES" head dimension (3 role logits) is unchanged while the agent logits drop from 7 to 3.

Paper file: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt` (2532 lines, fully read).

---

## EXTRACTION: OPTIMIZATION

I now have all optimization-relevant content extracted. I have everything needed to write the dense Markdown spec for the OPTIMIZATION dimension. Below is my return value.

---

# TRINITY — OPTIMIZATION Dimension Spec (sep-CMA-ES)

Source: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt`. All quotes cite line numbers in that file (the file has no printed page numbers beyond "Published as a conference paper at ICLR 2026"; I cite section headers + line ranges).

## 1. Algorithm choice & the regime it targets

> "We find that a derivative-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES) (Hansen et al., 2003) with diagonal covariance, separable CMA-ES (sep-CMA-ES) (Ros & Hansen, 2008), is effective in this particular regime: high dimensionality, weak parameter correlations, and high per-step cost." (lines 148-151)

> "We provide theoretical and empirical evidence that, in this extremely budget-tight scenario (1.5k–40k evaluations for a 10k-dimensional problem), sep-CMA-ES significantly outperforms RL and the random search baseline, suggesting strong block-ε-separability (see Definition 1) in the optimization objective." (lines 151-154)

So the headline regime numbers: **search dimension ≈ 10,000 (10k)**; **total evaluation budget range = 1.5k–40k evaluations** (lines 151-152). Verbatim mechanism description:

> "sep-CMA-ES, a black-box evolutionary strategy that iteratively improves a central 'parent' policy by sampling a population of perturbed parameter vectors, evaluating each candidate to obtain a fitness score, and recombining candidates via fitness-weighted averaging to form the next parent. Unlike full CMA-ES, sep-CMA-ES maintains only a diagonal covariance matrix, making the algorithm especially well suited to block-diagonal landscapes." (lines 346-350)

## 2. Search-space dimensionality and what is parameterized

Two trainable parameter sets total **< 20K parameters** (lines 176-177, 225). The optimization vector θ ∈ P ⊂ Rⁿ is the concatenation of:

1. **Lightweight head** appended after the SLM final hidden layer. Projects hidden state `h ∈ R^d` to `L + 3` logits — `L` logits for LLM selection, `3` logits for role (lines 217-219). Default head = **linear**, `z = Wh`, `W ∈ R^{n_a × d_h}`, **no bias**, exactly `d_h·n_a` params (lines 2123-2126).
2. **Singular-value fine-tuning (SVF)** of a subset of the SLM's weight matrices: SVD each selected matrix, learn only the singular-value scales, keep orthogonal matrices fixed (lines 220-224). Paper uses the **second-to-last layer** of Qwen3-0.6B for SVF (lines 380-381).

Backbone = **Qwen3-0.6B** as coordinator SLM (line 379); head = "a single linear layer of 10K parameters" (lines 379-380).

**Parameter sizes (Table 6, lines 2205-2209)** — computed for `d_h = 1024`, `n_a = 10`:

| component | params |
|---|---|
| SVF | 9216 |
| linear head | 10240 |
| low-rank head | 20680 |
| sparse head | 11266 |
| block-diagonal-2 head | 5120 |
| block-diagonal-10 head | 1024 |

So the **default optimized vector** = linear head (10240) + SVF (9216) = **19,456 parameters** (< 20K, consistent with lines 176-177, 225). The theory rounds this to `n ≈ 10000` (lines 233, 352, 1233, 1436). Note ambiguity below: the body says the head alone is "10K parameters" (line 380) and the theory uses `n ≈ 10000`, but Table 6 + SVF sum to ~19.5K.

**Action set:** `n_a = L + 3 = 10` (lines 218-219, 919-920) → **L = 7 LLMs in pool**, 3 roles. Confirmed: "na = 10 logits" with "one block per agent/role (B = 10)" (lines 2194-2195).

**Hidden dim:** `d_h = 1024` (Table 6 caption usage, lines 919, 2214).

## 3. Fitness / objective function

> `J(θ) := E_{τ∼π_θ}[R(τ)]` is the expected terminal reward (line 197). `R(τ) ∈ {0, 1}` is a **binary terminal reward** revealed at trajectory end (lines 195-196). Each end-to-end run = one "atomic evaluation, or a Bernoulli call since the rewards follow the Bernoulli distribution" (lines 200-201).

> "we seek θ⋆ ∈ arg max_θ∈P J(θ) under a tight atomic evaluation budget B_env that counts individual Bernoulli calls of the terminal reward used when estimating J(θ) (e.g., via replication/averaging)." (lines 202-204)

**Fitness = average binary task reward over `m` replications per candidate** (`m` = replication count; Bernoulli averaging, lines 200-204, 1248). **Maximized** (the theory works with `g := −J` for contraction analysis, line 1239). **No explicit cost/length penalty term is stated for the sep-CMA-ES fitness.** Output-token cost is constrained indirectly by a fixed 4096-token cap per LLM and max 5 turns (lines 402-404), not via a fitness penalty. Cost regularization is mentioned only for the MasRouter baseline (line 2392), not for TRINITY's own objective.

The policy: `π_θ(a | s) ∝ exp(f_θ(h(s))_a)`, softmax over agent–role logits (lines 190-192). Output conversion is **softmax by default**; **argmax** only for block-diagonal-10 (lines 927-928).

## 4. CMA-ES hyperparameters (verbatim)

- **Population size:** `λ = ⌈4 + 3 ln n⌉ (≥ 2)` (lines 354, 1246). At `n ≈ 10000`: "λ = ⌈4 + 3 ln n⌉ = ⌈4 + 3 ln 10000⌉ = 32" (line 1436). So **λ = 32**.
- **Parent count μ:** "µ parents with weights `(w_j)_{j=1}^µ`" (lines 1246-1247). **No numeric μ is given** — see OPEN section.
- **Replication counts:** `m_CMA = 16`, `m_RS = 32` (number of Bernoulli evaluations averaged per candidate) (lines 359, 1233, 1248, 1437).
- **Diagonal scaling / step:** `D_t = diag(√s_{1,t}, …, √s_{n,t}) ≻ 0`, sampling `y = m_t + σ_t D_t z`, `z ∼ N(0, I_n)` (lines 1242-1244). Whitened chart `x = D_t^{-1}(y − m_t) = σ_t z` (isotropic sampling) (line 1245).
- **Diagonal covariance learning rate:** `c_cov = Θ(1/n)` (lines 1413, 1484).
- **Per-iteration evaluation cost:** CMA uses `m_CMA · λ` evaluations per iteration ⇒ `T = ⌊B_env / (m_CMA λ)⌋` (lines 1423-1424, 1452).

**Initial σ (σ₀): NOT stated explicitly.** The theory references "step-size σ_t > 0" and a "local monotonicity window" (Assumption 2, lines 1242, 1295-1298) but gives no initial value. See OPEN.

## 5. Number of generations / total evaluation budget

- **CMA iteration range used in theory:** `T ∈ [2, 60]` (Proposition 1, line 1452) and `T ∈ [2, 60]` window (line 1452). Also "modest T (a few to a few dozen iterations)" (lines 1442-1443).
- **REINFORCE baseline ran for 60 iterations** (line 2232) — used to match CMA's evaluation count, implying **CMA was run ~60 iterations** in practice.
- **Total budget B_env:** stated only as the range **1.5k–40k evaluations** (line 152). Cross-check: at `λ=32, m_CMA=16, T=60` ⇒ `B_env = 16·32·60 = 30,720` evaluations (within the 1.5k–40k range). The exact T used for the reported runs is not pinned beyond "60 iterations" for the budget-matched baseline.
- **Budget matching to RS:** "budget matching across T CMA iterations yields N = (m_CMA λ / m_RS) T = (16·32/32) T ≈ ⌊16 T⌋" RS candidates (lines 1437-1440); and `N = ⌊B_env / m_RS⌋` (line 1300, 1424).

## 6. Normalization, bounds, restart logic

- **Normalization:** analysis done "in a covariance-normalized chart and mapped back through the current diagonal D_t" (lines 1227-1228). Features for the separability analysis are standardized (lines 1592-1593), but that is analysis-side, not the optimizer.
- **Bounds:** No explicit box constraints on θ for sep-CMA-ES. The only `[−0.5, 0.5]` band is for the **RS baseline** sampling range (line 2235), not CMA. The theory works in a "compact domain D ⊂ P" (line 1241) but gives no numeric bounds.
- **Restart logic:** **None stated.** No IPOP/BIPOP, no restart criterion mentioned anywhere.

## 7. Block-ε-separability theory (the claimed advantage)

**Definition 1 (Hessian-based block-ε separability in P)** (verbatim, lines 1256-1272):

> "There exists a structural diagonal scaling `S = diag(s_1, …, s_n) ≻ 0` such that the scaled Hessian `H_S(θ) := S^{1/2} H(θ) S^{1/2}` is uniformly nearly block-diagonal on D. With `D(θ) := diag(H_S(θ))`, one of the following dimensionless bounds holds with a common `ε_H ∈ [0,1)`:
> (B1) `sup_{θ∈D} ‖D(θ)^{-1/2} off_inter(H_S(θ)) D(θ)^{-1/2}‖_2 ≤ ε_H`
> (B2) `sup_{θ∈D} max_{i∈B_p, j∈B_q, p≠q} |[H_S(θ)]_{ij}| / √([H_S(θ)]_{ii}[H_S(θ)]_{jj}) ≤ ε_H`
> (B3) `sup_{θ∈D} max_{i∈B_p} (Σ_{j∈B_q, q≠p} |[H_S(θ)]_{ij}|) / [H_S(θ)]_{ii} ≤ ε_H (< 1)`.
> Within-block structure is unrestricted; `0 < µ_i ≤ [H_S(θ)]_{ii} ≤ L_i < ∞` on D." (Here `H(θ) := ∇²g(θ)`, `g := −J`; `off` zeroes the diagonal, `off_inter` zeroes diagonal and within-block entries — lines 1239, 1250-1251.)

**Supporting assumptions/definitions (verbatim):**
- **Assumption 1 (Diagonal comparability):** "There exist constants `c_cmp, C_cmp > 0` such that for all t, i, `c_cmp ≤ s_i / s_{i,t} ≤ C_cmp`, equivalently `C_cmp/c_cmp = O(sup_t κ_D(t))`." (lines 1274-1278)
- **Definition 2 (Metric–alignment factor):** `χ(u, D) := (u^⊤ D u)² / (u^⊤ D² u) ∈ [1/κ_D, 1]` (lines 1281-1284).
- **Assumption 2 (Local linear score with curvature remainder):** `J(m_t + σD_t z) = ½ + γσ⟨u_t,z⟩ + ξ_t(z)`, `|ξ_t(z)| ≤ (L_curv + c_H ε_H) σ² ‖z‖²` (lines 1295-1298).
- **Definition 3 (Rank attenuation under replication):** defines `ρ̃²_RS, ρ̃²_CMA ∈ [0,1]` (lines 1300-1312).
- **Assumption 3 (Metric–alignment comparability):** `1/C_χ ≤ [χ(u_t,D_t)/κ_D(t)] / [χ(u_0,D_0)/κ_D(0)] ≤ C_χ` (lines 1314-1317).
- `κ_D := s_max(D)²/s_min(D)²` (lines 1252-1255).

**Claimed advantage:**

> "for the small-T regime, Proposition 1 shows that sep-CMA-ES's improvement grows roughly linearly with the number of iterations, while random search (RS) grows only with the logarithm of how many candidates it can test. Thus, for modest T, sep-CMA-ES outperforms RS." (lines 356-358)

> "In the specific regime of our study (n ≈ 10000, λ ≈ 32, m_CMA = 16, m_RS = 32), budget matching yields about 16T RS candidates; the gain ratio behaves like `(T / ln(16T)) · η²`, where η is a reliability factor between 0 and 1, usually close to one. This ratio is greater than one even for small T." (lines 359-362)

> "Proposition 2 states that after about n iterations of calibration, sep-CMA-ES enters a steady regime where each step reduces the remaining error by a fraction of order 1/n, with a rate constant close to κ̄_{µ,λ}, where the constant κ̄_{µ,λ} = Θ(1) denotes the CMA recombination efficiency." (lines 363-365)

Constants: `α_{µ,λ} := E[⟨u_t, Σ_j w_j z_{j:λ}⟩]`, `β_{µ,λ} := E[‖Σ_j w_j z_{j:λ}‖²]`, `κ_{µ,λ} := α²_{µ,λ}/β_{µ,λ} = Θ(1/n)`, `κ̄_{µ,λ} := n·κ_{µ,λ} = Θ(1)` (lines 1396-1404).

**Proposition 1 (verbatim, lines 1452-1462):**
> "Fix `T ∈ [2, 60]` and let the CMA budget be `B_env = m_CMA λ T`. If the replication schedule ensures `ρ̃_CMA/ρ̃_RS ≥ η ∈ (0,1]` and the metric-alignment efficiency stays comparable across iterations (Assumption 3), then, up to an O(ε_H) term,
> `(CMA gain in J)/(RS gain in J) ≳ (κ̄_{µ,λ}/2) · [T / ln max{e, ⌊(m_CMA λ/m_RS)T⌋}] · η² − C / ln max{e, ⌊(m_CMA λ/m_RS)T⌋}`.
> The inequality holds for oracle step-sizes and, up to a universal constant factor, for fixed step-sizes within the local monotonicity window (Assumption 2)."

**Proposition 2 (verbatim, lines 1475-1482):**
> "Under Definition 1, Assumptions 1, 2, and 3, and a replication schedule with `ρ̃²_CMA = 1 − O(ε_H)`, sep-CMA-ES achieves, after a Θ(n) transient, the per-iteration contraction `(κ̄_{µ,λ}/n)(1 − O(ε_H))`, i.e., `E[r_T²] ≲ exp(−c′T/n) r_0²` for some c′ > 0 depending on κ̄_{µ,λ} and the residual O(ε_H). Restricting to diagonal covariances incurs only an O(ε_H) multiplicative loss relative to the block-diagonal optimum."

Key per-iteration gain bounds (verbatim): RS single-round gain eq.(1) lines 1385-1392; CMA per-iteration gain eq.(2) `E[r_t² − r_{t+1}²]/r_t² ≥ χ(u_t,D_t)·(1/κ_D(t))·κ_{µ,λ}ρ̃²_CMA − Cε_H` (lines 1406-1409); geometric-regime eq.(3) `E[r_{t+1}²|r_t] ≤ (1 − (κ̄_{µ,λ}/n) ρ̃²_CMA (1−O(ε_H))) r_t²` (lines 1416-1421); head-to-head ratio eq.(4) lines 1428-1430.

**Empirical separability evidence:** linear separability of hidden states by task type ≈ 0.993–0.997 (lines 893, 1673); head block-diagonal-10 "retains a large fraction of the performance despite its tiny parameter count" → "strong block-ε separability" (lines 338-341, 916-925).

## 8. Why REINFORCE is expected to fail (objective geometry)

> "we observe weak coupling among parameters — each has only a tiny influence on the scalar reward, making traditional methods like REINFORCE's per-parameter gradients low-SNR and therefore ineffective." (lines 144-146)

> "this geometry undermines the REINFORCE baseline … noisy global returns swamp weak inter-block signals, yielding ill-conditioned gradients, poor credit assignment, and unstable learning." (lines 343-345)

## 9. Baselines (RL / imitation / random search) — exact setups

**RS (random search with fitness averaging)** (lines 2229-2237):
> "For RS, we performed 32 trials for each sampled parameter vector, continuing until the total number of trials matched the evaluation count of sep-CMA-ES." (lines 2232-2233)
> "For RS, we warmstart it by calibrating the sampling range using the high-performing weights obtained via sep-CMA-ES. Specifically, we sample uniformly from `[−0.5, 0.5]`, a band that slightly exceeds the observed extrema of those weights. For each sampled parameter vector, we run 32 independent trials and compare the average reward." (lines 2234-2237)
- So RS: `m_RS = 32` Bernoulli trials per candidate, uniform sampling `θ ∼ U[−0.5, 0.5]^n`, total candidates `N = ⌊B_env/32⌋`, budget-matched to CMA. RS "often collapses to unipolar choices, over-selecting a single agent or role" (lines 955-956).

**REINFORCE (RL baseline)** (Williams, 1992; line 939):
> "For REINFORCE, we used a batch size equal to the per-iteration evaluation size of sep-CMA-ES and ran for 60 iterations." (lines 2231-2232)
- Per-iteration eval size of sep-CMA-ES = `m_CMA · λ = 16 · 32 = 512`, so REINFORCE batch size = 512, 60 iterations. Maintains "an almost uniform selection pattern" (line 954).

**SFT (imitation learning baseline)** (lines 1496-1539):
- Direct single-step state→action MLE; state = coordinator hidden representation, action = index of selected LLM (lines 1500-1505).
- Labels: per-question best-LLM via majority vote across seeds, uniform tie-break (lines 1506-1513).
- **Adam, frozen SLM, train only linear head; learning rate `1×10⁻⁶`, batch size 64** (lines 1534-1536).
- Label-cost argument why SFT can't scale to multi-turn: single-step needs `3 seeds × 7k datapoints × 7 agents = 147k` queries (lines 1552-1554); multi-turn (5 turns, 7 agents, 3 roles) blows up by `7⁴·3⁵ = 583,443 ≈ 5.8×10⁵`, total `≈ 8.7×10¹⁰` queries (lines 1555-1569).

**Table 4 — algorithm comparison, in-distribution, comparable budgets (lines 945-951):**

| Method | LiveCodeBench | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| REINFORCE | 0.253 | 0.459 | 0.500 | 0.266 |
| RS | 0.374 | 0.794 | 0.897 | 0.345 |
| SFT | 0.592 | 0.786 | 0.906 | 0.360 |
| **sep-CMA-ES** | **0.615** | **0.880** | **0.916** | **0.401** |

## 10. Head-architecture variants (the optimization vector's structure)

(Appendix A.4, lines 2119-2218; Table 3 lines 927-934.)
- **Linear (default):** `z = Wh`, `W ∈ R^{n_a×d_h}`, no bias, `d_h·n_a` params. Best overall (Table 3).
- **Low-rank:** `u = ELU(Uh)` with `ELU α = 0.1`; `z = Vu·σ`, `U∈R^{r×d_h}, V∈R^{n_a×r}`, fixed non-trainable scale σ, **bottleneck r = 14**; Xavier-uniform init with gains `U ∼ U[−√(6/(d_h+r)), √(6/(d_h+r))]`, `V ∼ U[−√(18/(r+n_a)), √(18/(r+n_a))]` (lines 2129-2154). 20680 params.
- **Sparse:** `z = W(h⊙α)`, learnable selection vector `α∈R^{d_h}`, `k = max(1, ⌊d_h·(1−sigmoid(ρ))⌋)`, learnable sparsity logit ρ; differentiable top-k via Gumbel(0,1) with **temperature τ ∈ [1.0, 20.0]**, hard top-k at inference; `d_h·n_a + d_h + 2` params (lines 2156-2172).
- **Block-diagonal-2:** B=2, proportional partition (lines 2183-2193). 5120 params.
- **Block-diagonal-10 + argmax:** B=10, `a_i=1`, hidden dims distributed evenly (`d_h/10 + 1` for `j ≤ d_h mod 10`, else `d_h/10`); 1024 params, exact 10× reduction vs linear; uses argmax output conversion to maximize independence (lines 2194-2203, 2213-2214).

## 11. Coordination loop / evaluation harness (each fitness eval)

- **Max turns `K = 5`** (lines 403-404); termination `τ = min{k ≤ K : R_k = V and u_k = ACCEPT}`, else `τ = K` (lines 330-331); final answer = `O_τ`.
- Per turn: select agent `A_k` from pool `M` (L=7) and role `R_k ∈ {Thinker(T), Worker(W), Verifier(V)}`; coordinator builds role-specific prompt from transcript `C_{k-1}`, queries `A_k`→`M_k`, post-processes to `O_k` (lines 304-321).
- Verifier outputs `u_k ∈ {ACCEPT, REVISE}` + optional diagnosis `δ_k` (lines 318-320).
- **Per-LLM max generated tokens = 4096, "minimal reasoning effort"** (lines 402-403).
- Coordinator reads the **penultimate output-token hidden state** (default) as head input (lines 273, 289-291); ablation uses last token (line 811).

## 12. Numbers cross-reference (all verbatim)

| Symbol | Value | Line(s) |
|---|---|---|
| n (head/search dim, theory) | ≈ 10000 | 152, 233, 352, 1233, 1436 |
| n (default optimized: linear+SVF) | 10240 + 9216 = 19,456 | Table 6, 2209 |
| d_h | 1024 | 919, 2214 |
| n_a (= L+3) | 10 | 218-219, 2194 |
| L (LLM pool) | 7 | 383-395 |
| roles | 3 (T/W/V) | 306, 309 |
| λ (population) | ⌈4+3 ln n⌉ = 32 | 354, 1246, 1436 |
| m_CMA | 16 | 359, 1437 |
| m_RS | 32 | 359, 1248, 1437 |
| per-iter evals (CMA) | m_CMA·λ = 512 | 1423-1424 |
| T (CMA iterations) | range [2,60]; 60 used for budget match | 1452, 2232 |
| B_env (total budget) | 1.5k–40k | 152 |
| c_cov | Θ(1/n) | 1413, 1484 |
| max turns K | 5 | 404 |
| max tokens/LLM | 4096 | 402 |
| RS sampling range | U[−0.5, 0.5] | 2235 |
| REINFORCE batch / iters | 512 / 60 | 2231-2232 |
| SFT optimizer / lr / batch | Adam / 1e-6 / 64 | 1534-1536 |
| low-rank r | 14 | 2137 |
| low-rank ELU α | 0.1 | 2133 |
| sparse temperature τ | [1.0, 20.0] | 2164 |

---

## OPEN / AMBIGUOUS

1. **Initial step-size σ₀ — ABSENT.** The paper never gives a numeric initial σ for sep-CMA-ES. Only "σ_t > 0" and a "local monotonicity window" (Assumption 2). You must pick a default (standard pycma `sigma0`) — not specified.
2. **Parent count μ — ABSENT (numeric).** Only "µ parents with weights `(w_j)_{j=1}^µ`" (line 1246). No value; standard sep-CMA-ES default is `μ = ⌊λ/2⌋ = 16` with default log-weights, but the paper does not state it.
3. **Recombination weights `w_j` — unspecified.** Referenced as `(w_j)` and used in `Σ w_j z_{j:λ}`; not given. Assume CMA-ES default weights unless the authors deviate.
4. **Exact total iterations / B_env for the reported main runs — ambiguous.** Range "1.5k–40k evaluations" (per-task differs); only the budget-matched baseline run states "60 iterations." Whether all four in-distribution tasks used T=60 (B_env=30,720) or different T per task is not stated.
5. **n: 10K vs 19.5K mismatch.** Body calls the head "10K parameters" and theory uses `n ≈ 10000`, but Table 6 head(10240)+SVF(9216) = 19,456. Unclear whether sep-CMA-ES optimizes head-only (n≈10240) or head+SVF jointly (n≈19,456). Figure-2/Section-3.1 imply both head and SVF are trainable; the "<20K" total supports joint optimization. Needs a decision.
6. **SVF specifics — partially absent.** "Second-to-last layer of the 0.6B model" is selected (line 381), but *which* weight matrices within that layer (q/k/v/o/MLP), and the SVF init (singular-value scales initialized to 1.0?), are not stated. 9216 SVF params is the only quantitative anchor.
7. **Reward `R(τ)` definition per task — not specified at the optimizer level.** It is "binary terminal reward ∈ {0,1}"; the exact correctness checker per benchmark (e.g., LiveCodeBench test-pass, MATH500 answer-match, MMLU/RLPR scoring) is not given in the optimization sections.
8. **Replication-fitness aggregation detail.** Fitness = average over `m_CMA = 16` Bernoulli draws, but whether the 16 draws are 16 different task instances, 16 seeds on the same instance, or a mixed minibatch is not stated. The SFT label section uses "3 seeds" (line 1553); RS uses "32 independent trials" (line 2236). The minibatch/seed construction for CMA fitness is ambiguous.
9. **Bounds on θ for CMA — none.** No box constraints stated for the CMA search (the [−0.5, 0.5] band is RS-only). Whether θ is clipped/normalized during CMA is unspecified.
10. **No restart logic** of any kind is described (no IPOP/BIPOP, no stagnation criterion). Treat as single-run sep-CMA-ES.
11. **Role-specific prompt templates — NOT provided verbatim.** Section 3.2 describes role contracts (Thinker/Worker/Verifier) and the Figure-1 example shows paraphrased messages (lines 76-122), but no literal system/role prompt strings are given. The "Msg Processing"/"message processing module injects a role-specific prompt" (lines 127, 307) is described only functionally.
12. **CMA-ES variant constants** (`c_σ`, `d_σ`, damping, `c_1`/`c_μ`) beyond `c_cov = Θ(1/n)` are not given — use Ros & Hansen (2008) sep-CMA-ES defaults.
13. **Model pool vs. replication context mismatch (informational, not a paper gap):** the paper's pool is GPT-5, Gemini-2.5-pro, Claude-4-Sonnet, Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B (direct), Qwen3-32B (reasoning) (lines 384-395), i.e. L=7. Your replication uses 3 Fireworks models (deepseek-v4-pro, glm-5p2, kimi-k2p6), which changes L=3 ⇒ n_a = L+3 = 6 ⇒ linear head = 6·1024 = 6144 params, λ = ⌈4+3 ln n⌉ recomputed on the new n. This is a deliberate deviation from the paper, not an omission in it.

---

## EXTRACTION: EXPERIMENTS

I now have the complete paper. Below is my dense, fully-cited extraction for the EXPERIMENTS dimension.

---

# TRINITY (ICLR 2026) — EXPERIMENTAL SETUP, verbatim extraction

Scope: model pool, benchmarks/datasets, splits/sizes, metrics, evaluation protocol, decoding settings, seeds, compute, API/token budgets, and baselines. Every number, hyperparameter, dimension, token budget, and template is quoted with its source location.

---

## 1. Model pool (the LLMs being coordinated)

From Section 4.1 "Coordinator and agents" (p.5):

> "Our model pool contains **seven models** from both open-source communities and closed-source API providers. These are, three top-tier closed-source models currently available (**GPT-5** (OpenAI, 2025), **Gemini-2.5-pro** (Comanici et al., 2025), and **Claude-4-Sonnet** (Anthropic, 2025)), and four well-known open-source models (**Gemma-3-27B-It** (Team et al., 2025), **DeepSeek-R1-Distill-Qwen-32B** (Guo et al., 2025), **Qwen-3-32B (direct)**, and **Qwen-3-32B (reasoning)**)."

Exact agent-ID mapping (Figure 15 caption, p.28):
- **A0: GPT-5**
- **A1: Claude-Sonnet-4-20250514**
- **A2: Gemini-2.5-pro**
- **A3: DeepSeek-R1-Distill-Qwen-32B**
- **A4: Gemma-3-27b-It**
- **A5: Qwen3-32B (reasoning)**
- **A6: Qwen/Qwen3-32B (direct)**

Note for replication context: Qwen3-32B appears twice, as two distinct "agents" (direct vs reasoning mode of the same checkpoint). So L = 7 agents but only ~6 distinct checkpoints. The specific Claude pin is `claude-sonnet-4-20250514`. The paper does NOT pin GPT-5 / Gemini-2.5-pro version dates, nor Gemma-3-27B-It / DeepSeek-R1-Distill-Qwen-32B revisions beyond names. (Your replication uses Fireworks `deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6` — a completely different pool; the paper's pool is NOT these models.)

### Coordinator SLM (run locally on your H200)
From Section 4.1 (p.5):
> "We use **Qwen3-0.6B** (Yang et al., 2025) as the coordinator's SLM, paired with a single **linear layer of 10K parameters** as the simple but effective head, and select the **second-to-last layer** of the 0.6B model for singular value fine-tuning."

Total learnable parameters: "**under 20K**" (Abstract; Section 1; Section 3.1). Head ≈ 10K; SLM ≈ 0.6B params; head "≈ 10K parameters" (Abstract).

Head input: penultimate output token's hidden state, h ∈ R^d, d = 1024 (Section 4.6 / Appendix A.3 state "1024 dimensions"; Figure 2 caption). Head output size = L + 3 = 10 logits (7 agent logits + 3 role logits) — "projects a hidden state h ∈ R^d to an output of size L + 3" (Section 3.1).

Parameter sizes (Table 6, p.25), computed on Qwen3-0.6B with d_h=1024, n_a=10:
| SVF | linear | low-rank | sparse | block-diag-2 | block-diag-10 |
|-----|--------|----------|--------|--------------|----------------|
| 9216 | 10240 | 20680 | 11266 | 5120 | 1024 |

(SVF = singular value fine-tuning = 9216 params; the default head used in main results is `linear` = 10240.)

---

## 2. Benchmarks and datasets

### In-distribution (training + evaluation): four tasks
From Section 1 and Section 4.1 "Tasks and protocols" (p.5):
> "We train and evaluate TRINITY across four diverse tasks, including **MATH500**, **MMLU**, **RLPR**, and **LiveCodeBench**."

Citations: Math500 = Lightman et al., 2023; MMLU = Hendrycks et al., 2020; RLPR = Yu et al., 2025 (arXiv:2506.18254, "Extrapolating RLVR to general domains without verifiers"); LiveCodeBench = Jain et al., 2024.

> "For each task, we train on the designated training set and assess performance on the corresponding test set, utilizing **official splits where available**."

**LiveCodeBench split — exact (Section 4.1, p.5):**
> "For LiveCodeBench specifically, we use the **V1 release (400 samples)** for training and conduct evaluation on the **newly introduced questions in the V6 release (175 samples)**."

Time window for the V6/test questions (Abstract & Section 4.4): "**LiveCodeBench (Jan - Aprl 2025)**" / "newly-released questions spanning **January to April 2025**."

### Out-of-distribution (held-out, zero-shot transfer): four tasks
From Section 1 and Section 4.1 (p.5):
> "...four challenging held-out tasks (**AIME2025, BigCodeBench, MT-Bench, and GPQA-D**)..."

Citations: AIME = Veeraboina, 2023 (Kaggle "AIME problem set 1983-2024"); BigCodeBench = Zhuo et al., 2024; MT-Bench = Bai et al., 2024 (this reference is actually **MT-Bench-101**, arXiv:2402.14762); GPQA-D (Diamond) = Rein et al., 2024.

### Dataset sizes used in SFT label generation (Appendix A.2.2, p.19)
> "...generating labels requires running **3 seeds on 7k datapoints across 7 agents**, resulting in 3 × 7k × 7 = **147k LLM queries**."

So the combined in-distribution training pool ≈ **7,000 datapoints**, evaluated with **3 seeds**. Per-task best-agent label counts (Table 5, p.18) imply roughly: LiveCodeBench ≈ 175 best-labels (sum of counts 31+69+31+13+6+19+6 = 175 — matches the V6 test size, so Table 5 is on the eval set), MATH500 ≈ 100 (18+13+21+12+15+17+4 = 100), MMLU ≈ 1,531, RLPR ≈ 5,184 (sum), Overall ≈ 6,990 (≈ "7k"). MATH500 evaluated on 100 samples here (not the full 500) per the Table 5 counts.

The synthetic separability study (Appendix A.3) replicates: "**1024 dimensions, 7 agent classes, 4 task type clusters**."

### OPEN / AMBIGUOUS — datasets/splits
- The paper says "official splits where available" but gives **explicit train/test sizes only for LiveCodeBench** (400 train / 175 test). For MATH500, MMLU, RLPR it does **not** state how many samples were used for training vs evaluation, nor the train/test split ratios. Table 5's per-task counts (MATH500=100, MMLU=1531, RLPR=5184) appear to be evaluation/label sets, not training sizes — this is **ambiguous**.
- No explicit **validation** split is described for TRINITY itself (early stopping / model selection criterion for the CMA-ES run is not stated). MasRouter/RouterDC baselines mention validation, but TRINITY's own validation protocol is absent.
- AIME held-out set is called both "AIME" (Table 1) and "**AIME2025**" (Section 4.1). The size of the AIME2025 held-out set is not given (AIME typically = 30 problems/year). BigCodeBench/MT-Bench/GPQA-D held-out sizes are not stated.
- GPQA-D = GPQA-Diamond is implied (not spelled out).
- LiveCodeBench V1 "400 samples" vs the public LCB release versions — the paper does not give a code-generation subset definition beyond "V1 release (400 samples)" and "V6 release (175 newly introduced)".

---

## 3. Metrics

- **LiveCodeBench: pass@1.** "achieving a pass@1 of **86.2 ± 0.5%**" (Abstract); "**0.862 ± 0.005**" and "**0.838±0.007** (GPT-5), **0.672±0.031** (Gemini-2.5-Pro), **0.465±0.010** (Claude-4-Sonnet)" (Figure 4, p.7). In-distribution LCB v6 main result "**0.61 pass@1**" (Section 4.2).
- **MATH500, MMLU, RLPR, BigCodeBench, GPQA-D:** accuracy (fraction correct / pass@1-style; reported as decimals 0–1 in Figure 3 and Tables 2–4, or percentages in Table 1).
- **MT-Bench:** score on a ~10-point scale (e.g., 9.60 for TRINITY, Table 1; 9.37 for Gemini), i.e., LLM-judge rating, not accuracy.
- **Terminal reward R(τ) ∈ {0,1}** — binary Bernoulli reward per atomic evaluation (Section 2). Objective J(θ) = E_{τ∼π_θ}[R(τ)].
- **Mean relative error reduction**: "**21.9%** over the second-best approach" (Abstract, across the 4 in-distribution benchmarks); "**11.76%** relative error reduction on MATH500 compared to the 2nd best method (Gemini Pro 2.5 with 5x CTX)" (Section 4.2). RER formula defined in Appendix A.6, Eq.14: RER = (Z_{C,M'} − S*_{C,M'}) / (1 − S*_{C,M'}).
- "**Per-Question-Best**" upper bound = "the optimal performance achievable by correctly selecting the best-performing worker model for each individual question" / "union of all correct answers from the seven LLMs" (Section 4.2/4.3; Appendix A.7.1).

### Seeds / error bars
- LiveCodeBench full-power results report ± std over seeds (e.g., 0.862±0.005). Number of seeds for these bars is **not explicitly stated** in the main text, but the SFT label-generation section uses "**3 seeds**" (Appendix A.2.2), and Table 5 / Appendix A.2.1 ("for each seed independently... aggregate across all seeds via majority voting") confirm **3 seeds** for the per-question-best oracle. MMLU Majority@5 baselines (Table 7) report ±std (e.g., 91.57±0.70) — again seed count not specified there.

### OPEN / AMBIGUOUS — metrics/seeds
- **Number of evaluation seeds for the main TRINITY results** (Figure 3, Tables 1–4) is not stated explicitly; only the SFT oracle uses "3 seeds." Whether the ±0.005-type bars come from 3 seeds or another count is **ambiguous**.
- pass@1 sampling method for LCB (greedy vs sampled@temperature) is not stated.
- For MT-Bench, the judge model and scoring rubric are not specified.

---

## 4. Evaluation protocol & decoding settings

From Section 4.1 "Tasks and protocols" (p.5):
> "To ensure consistency between open and closed models and facilitate training, we set the **default maximum generated tokens to 4096** for each LLM, with **minimal reasoning effort**. We also set the **maximum number of coordination turns to five**."

Coordination protocol (Section 3.2):
- At most **K = 5** turns per query.
- Three roles: **Thinker (T), Worker (W), Verifier (V)**.
- Termination: τ = min{ k ≤ K : R_k = V and u_k = ACCEPT }, with τ = K if no acceptance. Verifier outputs u_k ∈ {ACCEPT, REVISE} plus optional diagnosis δ_k. Final answer = O_τ.
- Transcript construction: input to agent A_k = original query Q concatenated with full transcript C_{k-1} = (Q, O_1, …, O_{k-1}); the coordinator injects a "brief, role-specific prompt" before querying, and "lightly post-processes M_k into O_k."
- The coordinator's own generated **text is discarded**; only the head's logit outputs are used (Section 3.1). Head can read an earlier (penultimate) token hidden state for rapid inference rather than full generation.

**Turn-budget ablation (Figure 4 bottom, p.7):** LiveCodeBench pass@1 improves "from **0.823 to 0.863** as turns increase **from 2 to 6**." (max turns swept at 2, 4, 6.)

**Full-power LiveCodeBench setting (Section 4.4, p.7):**
> "...we limited the maximum output length for all LLMs... For the LiveCodeBench task, the coordinator's LLM selection narrows down to the **three closed-models** after training. This allows us to **remove the output length constraint**... **we simply remove the constraints and do not retrain TRINITY.**"

Decoding: "minimal reasoning effort" (4K token cap) is the default for all in-distribution runs. No temperature, top-p, or top-k value is given anywhere in the paper.

### OPEN / AMBIGUOUS — decoding/protocol
- **No sampling temperature, top-p, top-k, or repetition penalty is stated** for any model (coordinated LLMs or coordinator SLM). "minimal reasoning effort" is the only decoding qualifier — undefined numerically.
- The exact **role-specific prompt templates** ("brief, role-specific prompt") are **never quoted** in the paper. Roles are described functionally (Thinker strategizes / Worker executes / Verifier evaluates with ACCEPT|REVISE + diagnosis) but no verbatim prompt strings, message-processing rules, or post-processing ("lightly post-processes M_k into O_k") logic are given. This is a major implementation gap.
- The "message processing module" that injects prompts is referenced (Figure 1) but its mechanics are not specified.
- For the OOD/held-out tasks, whether the 4096-token cap and 5-turn budget still apply is not restated (assumed same since "zero-shot transfer... do not retrain").

---

## 5. CMA-ES / training hyperparameters (relevant to compute budget)

From Section 3.3, Appendix A.1, A.5 (run locally on your H200):
- Algorithm: **sep-CMA-ES** (separable / diagonal-covariance CMA-ES; Ros & Hansen, 2008).
- Head dimension n ≈ **10000** (10K params).
- Population size: **λ = ⌈4 + 3 ln n⌉ = ⌈4 + 3 ln 10000⌉ = 32** (Section 3.3; Appendix A.1.2 "Trinity-scale instantiation").
- Replication counts: **m_CMA = 16**, **m_RS = 32** (evaluations per candidate) (Section 3.3; A.1).
- Evaluation budget regime: "**1.5k–40k evaluations for a 10k-dimensional problem**" (Section 1). Benv counts individual Bernoulli (terminal-reward) calls.
- Optimization iterations T analyzed for **T ∈ [2, 60]** (Proposition 1); REINFORCE and RS run to match budget.
- REINFORCE baseline (Appendix A.5): "**batch size equal to the per-iteration evaluation size of sep-CMA-ES** and ran for **60 iterations**."
- RS baseline (Appendix A.5): "**32 trials for each sampled parameter vector**, continuing until the total number of trials matched the evaluation count of sep-CMA-ES." Warmstart: sample uniformly from **[−0.5, 0.5]**.
- SFT baseline (Appendix A.2.1): Adam (Kingma & Ba, 2017), frozen SLM, train only linear head, **learning rate 1×10⁻⁶**, **batch size 64**.
- Low-rank head bottleneck **r = 14**, ELU α = 0.1, fixed scale σ (Appendix A.4). Sparse head Gumbel-softmax temperature **τ ∈ [1.0, 20.0]**. Block-diagonal-10 uses **argmax** output conversion (default elsewhere is softmax).

### OPEN / AMBIGUOUS — CMA-ES
- **σ (initial step size)**, **number of parents μ and recombination weights w_j**, **initial mean/parameter init**, and **total iteration count T actually used to train** the reported models are not given (only the analyzed range T∈[2,60] and λ=32, m_CMA=16). The "1.5k–40k evaluations" range is given but not the exact per-task value.
- Which **subset of SLM weight matrices** receives singular-value fine-tuning beyond "second-to-last layer" (which matrices in that layer) is not fully specified.

---

## 6. Compute budget & API/token budgets

### Hardware
- "Due to **hardware constraints in serving open-source models**, we limited the maximum output length for all LLMs in the pool for fair comparisons" (Section 4.4). No GPU count, GPU type, or wall-clock training time is given anywhere. (Your replication: single H200, GPU index 5, 143GB — paper gives no comparable hardware spec.)

### API/LLM-query budgets
- **Per atomic evaluation (one trajectory τ):** up to **5 LLM calls** (one per turn, K=5), each capped at **4096 output tokens** by default. This is the per-evaluation API-call/token budget.
- **SFT single-step label generation cost:** "3 × 7k × 7 = **147k LLM queries**" (= 1.5×10⁵) (Appendix A.2.2).
- **Multi-turn SFT label complexity (why SFT is infeasible):** multiplicative factor "**7⁴ · 3⁵ = 583,443 ≈ 5.8×10⁵**", inflating to "**1.5×10⁵ × 5.8×10⁵ ≈ 8.7×10¹⁰ LLM queries**" (Appendix A.2.2). (This is the cost SFT would incur, contrasted against label-free sep-CMA-ES.)
- **MoA baseline cost:** "**8 model calls per question**" (single MoA layer + single aggregator) (Appendix A.7.1).
- **RouterDC training:** "**500 samples** from the MMRL dataset... Each sample is repeated **5 times**" (Appendix A.7.1).
- **MasRouter training:** "**MMRL dataset with 256 samples, validating every 5 epochs**" (Appendix A.7.1).

### Output-token usage (efficiency, Appendix A.7.4)
Average output tokens per method on in-distribution tasks (Table 9, p.29):
| Model | Math500 | MMLU | RLPR | LiveCodeBench |
|---|---|---|---|---|
| **TRINITY** | 2,853 | 1,200 | 2,141 | 1,999 |
| MOA | 6,871 | 5,218 | 11,086 | 21,634 |
| RouterDC | 624 | 374 | 811 | 1,552 |
| Smoothie | 6,472 | 4,718 | 10,580 | 17,864 |
| MASRouter | 4,260 | 1,847 | 5,370 | 8,401 |

Per-model average output tokens are tabulated for 5× Self-Reflection (Table 10), 5× Context (Table 11), and Default Context = 4096 (Table 12). Examples (Table 12, Default 4096-cap): GPT-5 LCB=1,113; Gemini-2.5-Pro LCB=2,396; DeepSeek-R1-Distill-32B LCB=3,443. Full tables present for all 7 models × 4 tasks.

### OPEN / AMBIGUOUS — compute
- **No total GPU-hours, no wall-clock, no GPU model/count** for the CMA-ES run or coordinator inference.
- **No dollar API cost or total token consumption** for the full training run (only per-evaluation and the hypothetical SFT label costs are given).
- The exact **Benv used per benchmark** (within the 1.5k–40k range) is not specified.

---

## 7. Baselines compared

From Section 4.1 "Baselines" and Appendix A.7.1 / A.7.3:

**Multi-agent routing / ensemble / coordination methods:**
- **MasRouter** (Yue et al., 2025) — trained on same dataset, cost-regularization, MMRL 256 samples, validate every 5 epochs, best checkpoint after overfitting evidence.
- **RouterDC** (Chen et al., 2024) — trained on 500 MMRL samples, each repeated 5×, avg-performance label.
- **Smoothie** (Guha et al., 2024) — test-time; evaluated under both **dependent** (one agent per question) and **independent** (one agent for whole test set) strategies.
- **MoA / Mixture-of-Agents** (Wang et al., 2024) — single MoA layer + single aggregator, 8 calls/question, random model as final aggregator.
- **Random agent selection** — random agent each turn, max 5 turns (same as TRINITY).

**Single-model baselines (the 3 strongest in the pool):**
- **GPT-5, Gemini-2.5-pro, Claude-4-Sonnet**, each at **4K** and **20K (= 5× CTX)** max inference tokens (the 5× matches the 5-turn budget for fairness).
- **5× Self-Reflection (5x SR):** single agent, initial answer + reflection over 5 turns.

**Additional baselines:**
- **Majority voting / Majority@5** (5 samples per question) — only for discrete/multiple-choice (MMLU). Table 7: Gemini Pro 2.5 91.57±0.70, GPT-5 91.31±0.23, Claude-4-Sonnet 90.99±0.39.
- **LLM as Coordinator** (Appendix A.7.3): **Gemini Pro 2.5** prompted to pick model+role each turn. Underperforms TRINITY: **64.14 vs 70.44** average (Section 4.1 text says 64.14; Table 8 shows Gemini-as-coordinator avg **53.76** — see ambiguity below). Table 8: Math500 78.67, MMLU 83.26, RLPR 26.83, LiveCodeBench 26.28 → avg **53.76**, vs TRINITY 88.00/91.56/40.72/61.49 → avg **70.44**.
- **Per-Question-Best** — theoretical upper bound (argmax over 7 models per question).

**Learning-algorithm baselines (Table 4, Section 4.8):** sep-CMA-ES vs **REINFORCE** (Williams, 1992) vs **SFT** vs **RS** (random search with fitness averaging). Results (LCB / MATH500 / MMLU / RLPR): REINFORCE 0.253/0.459/0.500/0.266; RS 0.374/0.794/0.897/0.345; SFT 0.592/0.786/0.906/0.360; sep-CMA-ES 0.615/0.880/0.916/0.401.

### OPEN / AMBIGUOUS — baselines
- The "LLM as Coordinator underperforms... **64.14 vs 70.44**" in Section 4.1 text **conflicts** with Table 8's **53.76** for Gemini-as-coordinator (70.44 matches TRINITY in both). The 64.14 figure is not reconciled anywhere — likely a different configuration; **ambiguous which value is correct**.
- "MMRL dataset" used to train MasRouter/RouterDC is named but not defined/cited in the paper — its composition relative to the 4 in-distribution tasks is unclear.

---

## 8. Key result tables (for replication targets)

**In-distribution (Figure 3, decimals):** TRINITY ≈ Math500 0.88, MMLU 0.92(0.916), LCB-v6 0.61, RLPR 0.41 (exact values in Table 3 `linear` row: LCB 0.615, MATH500 0.880, MMLU 0.916, RLPR 0.401).

**Ablations (Table 2, p.6):** TRINITY avg 70.44; w/o SVF 67.85; w/o Thinker-role 68.69; w/o Tri-role 67.02; w/ Last token 64.66; Claude-4-Sonnet only 61.12; Gemini Pro 2.5 only 62.99; GPT-5 only 65.95. Columns: LiveCodeBench / MATH500 / MMLU / RLPR / Average.

**Held-out / OOD (Table 1, p.6):** TRINITY AIME 50.00, BigCodeBench 35.80, MT-Bench 9.60, GPQA-D 76.82, Average 54.21 — beats every single model (best single = Gemini Pro 2.5 avg 52.34).

---

## 9. Source file

Paper text read in full: **`/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt`** (2532 lines; arXiv:2512.04695v3).

---

## CONSOLIDATED OPEN / AMBIGUOUS (implementation blockers for the EXPERIMENTS dimension)

1. **Role prompt templates are never quoted** — Thinker/Worker/Verifier "brief, role-specific prompts," message-processing injection, and `M_k → O_k` post-processing logic are all undescribed verbatim. Largest gap.
2. **No decoding parameters** (temperature/top-p/top-k) for any model; "minimal reasoning effort" undefined numerically.
3. **Train/test/validation sizes given explicitly only for LiveCodeBench** (400 train V1 / 175 test V6). MATH500, MMLU, RLPR train sizes and split ratios unspecified; no TRINITY validation/early-stopping protocol stated.
4. **Held-out (OOD) set sizes** for AIME2025, BigCodeBench, MT-Bench, GPQA-D not given; MT-Bench judge/rubric unspecified.
5. **Number of evaluation seeds** for main results not stated (only SFT oracle uses 3 seeds); source of ±std bars unclear.
6. **CMA-ES details incomplete:** σ_init, μ, recombination weights, init scheme, and the exact total iteration count / Benv per task are not given (only λ=32, m_CMA=16, m_RS=32, n≈10k, budget range 1.5k–40k).
7. **No compute reporting:** no GPU count/type, no GPU-hours, no wall-clock, no total API token/cost for the actual training run.
8. **LLM-as-Coordinator score conflict:** Section 4.1 says 64.14; Table 8 says 53.76.
9. **"MMRL dataset"** (for MasRouter/RouterDC) named but undefined/uncited.
10. **Model version pins missing** for GPT-5, Gemini-2.5-pro, Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B (only Claude is pinned: `claude-sonnet-4-20250514`). Note the paper's pool is entirely different from your Fireworks pool (deepseek-v4-pro / glm-5p2 / kimi-k2p6).

---

## EXTRACTION: RESULTS

I have the complete paper. Now I'll compile the results-focused extraction. Let me also note the Figure 3 numbers, which are dense but readable from the OCR of lines 415-454.

Below is the complete results extraction.

---

# TRINITY — RESULTS DIMENSION: Target Metrics for Replication

All numbers quoted verbatim with section/page cites. Paper: "TRINITY: An Evolved LLM Coordinator," ICLR 2026 (arXiv:2512.04695v3). File: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt`.

## 0. Headline claims (Abstract + §1, p.1–2)

- "TRINITY achieves state-of-the-art performance, including a new record of **86.2%** on LiveCodeBench" (Abstract, p.1). Exact: "**pass@1 of 86.2 ± 0.5%**" (§1, p.2). Also written "**0.862±0.005**" (Fig.4, p.7).
- "a mean **relative error reduction of 21.9%** over the second-best approach" (§1, p.2) — across the four in-distribution benchmarks.
- "outperforms all single-model baselines with fair, adjusted output-token budgets" (§1, p.2).
- Zero-shot transfer to 4 unseen tasks "with performance surpassing each of the single models it orchestrates" (§1, p.2).

These are the RELATIVE claims to reproduce regardless of our different model pool:
1. **TRINITY > best single model** (with 5x-budget-matched single models) on every in-distribution and held-out task.
2. **TRINITY > best multi-agent baseline** (MoA, MasRouter, RouterDC, Smoothie, random routing).
3. **TRINITY > random routing** (random agent selection).
4. **TRINITY ≈ Per-Question-Best upper bound** on 3 of 4 in-distribution tasks.
5. **sep-CMA-ES > SFT > RS > REINFORCE** as the coordinator-training algorithm.

---

## 1. The 86.2% LiveCodeBench record — config that produced it (§4.4 "Unleashing Full Power," p.7; Fig.4)

- **Score: 0.862 ± 0.005 pass@1** on **LiveCodeBench V6** (newly released questions, **January–April 2025**, 175 samples).
- Config: "For the LiveCodeBench task, the coordinator's LLM selection narrows down to the three closed-models after training. This allows us to **remove the output length constraint**... we simply **remove the constraints and do not retrain** TRINITY." (§4.4, p.7). So: same trained coordinator as §4.2, but the 4096-token cap is lifted (vs. the constrained 0.61 number).
- Beats constituent models (Fig.4 top, p.7): **GPT-5 = 0.838 ± 0.007**, **Gemini 2.5-Pro = 0.672 ± 0.031**, **Claude-4-Sonnet = 0.465 ± 0.010**.
- Max-turns scaling (Fig.4 bottom, p.7): "improving from **0.823 to 0.863** as turns increase from **2 to 6**" (i.e., ~0.823 @ 2 turns → ~0.863 @ 6 turns). The constrained main-table number uses max 5 turns.

---

## 2. In-distribution evaluation (§4.2, Fig.3, p.6) — constrained 4096-token setting, max 5 turns

TRINITY's own scores appear in two places with slightly different precision. The ablation/algorithm tables (Tables 2–4, all "linear" head) give the canonical TRINITY row:

| Task | TRINITY | Source |
|---|---|---|
| LiveCodeBench (V6, constrained) | **61.46** (also 0.615 / 0.61) | Table 2 / Table 3 / §4.2 |
| MATH500 | **88.00** | Table 2 |
| MMLU | **91.56** | Table 2 (Table 8 confirms 91.56) |
| RLPR | **40.72** | Table 2 (0.401 in Table 3) |
| **Average** | **70.44** | Table 2 |

Explicit relative-improvement claims (§4.2, p.6):
- LiveCodeBench v6: "**0.61 pass@1**... substantially surpassing all competing methods."
- MATH500: "**11.76% relative error reduction** ... compared to the 2nd best method (**Gemini Pro 2.5 with 5x CTX**)."

### Figure 3 baseline numbers (OCR, p.6) — per benchmark
These are the per-bar values printed in Fig.3 (boldface "Trinity (Deepseek)" is TRINITY; "Per-Question-Best" is the upper bound). The x-axis groups include single models at 4 / 5x CTX / 5x SR, the multi-agent methods, random, Per-Question-Best, and TRINITY. OCR is partially garbled; values that are clearly legible:

- **Math500 row** (legible cluster): 0.85, 0.86, 0.82, 0.83, 0.82, 0.84, 0.74, 0.76, 0.77, 0.72, 0.83, 0.77, 0.77, 0.59, **0.88 / 0.91** (the 0.88 = TRINITY, 0.91 = Per-Question-Best per §4.2 "close to Per-Question-Best").
- **MMLU row**: 0.92, 0.92, 0.88, 0.91, 0.87, 0.91, 0.90, 0.89, 0.92, 0.84, 0.88, 0.88, 0.83, 0.84, **0.92** (TRINITY ≈ 0.92; Per-Question-Best 0.97 printed top-right).
- **LiveCodeBench row**: ... 0.53, 0.47, 0.57, 0.58, 0.58, 0.61 (TRINITY = **0.61**), 0.65 (Per-Question-Best), 0.25; with another 0.61 cluster.
- **RLPR row**: 0.40, 0.40, 0.39, 0.38, 0.34, 0.34, 0.41, 0.33, 0.32, 0.32, 0.33, 0.33, 0.32, 0.32, 0.28, 0.38, 0.38, 0.35, 0.39, 0.35, 0.31, 0.31, 0.25. MoA = 0.38, RouterDC = **0.28**, random = **0.32**, TRINITY ≈ 0.41.

Cited per-baseline anchors (§4.2 text, p.6, unambiguous):
- **MoA**: Math500 = **0.83**, RLPR = **0.38**, LiveCodeBench = **0.39**.
- **RouterDC RLPR = 0.28** vs **random selection 0.32** ("collaboration approaches even degrade performance below random").

> NOTE for replication: Fig.3 lacks a clean machine-readable table for every baseline cell — see OPEN section. The reliable single-model in-distribution numbers come from Table 2 ("only" rows) and Table 8.

### Single-model-only rows (in-distribution, from Table 2, p.6)
| Model | LiveCodeBench | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| Claude-4-Sonnet only | 39.09 | 82.25 | 88.23 | 34.90 |
| Gemini Pro 2.5 only | 46.51 | 83.05 | 79.41 | 43.00 |
| GPT-5 only | 59.54 | 75.66 | 90.74 | 37.87 |

(These let us verify TRINITY 70.44 avg > each single model, and that no single model dominates all four tasks — the core RELATIVE claim.)

---

## 3. Hold-out / zero-shot transfer (Table 1, §4.3, p.7)

Score scale: AIME, BigCodeBench, GPQA-D as percentages; MT-Bench on a ~10-point scale; "Average" is the paper's reported column.

| Model | AIME | BigCodeBench | MT-Bench | GPQA-D | Average |
|---|---|---|---|---|---|
| Gemini Pro 2.5 | 46.67 | 35.10 | 9.37 | 75.25 | 52.34 |
| GPT-5 | 46.67 | 33.80 | 9.35 | 72.73 | 51.07 |
| Claude-4-Sonnet | 35.33 | 35.80 | 9.28 | 67.30 | 46.14 |
| Qwen3-32B (reasoning) | 23.33 | 20.90 | 8.99 | 59.09 | 34.44 |
| DeepSeek-R1-Qwen-32B | 30.00 | 24.30 | 8.43 | 51.01 | 35.10 |
| Qwen3-32B (direct) | 20.00 | 23.00 | 9.03 | 54.05 | 33.46 |
| Gemma-3-27B-IT | 20.00 | 20.30 | 8.76 | 33.33 | 21.38 |
| **TRINITY (Ours)** | **50.00** | **35.80** | **9.60** | **76.82** | **54.21** |

Claim (§4.3, p.7): TRINITY "achieves the highest average score (54.21) and outperforms every individual baseline on each of the four tasks. It secures top performance on AIME (50.00), MT-Bench (9.60) and GPQA-D (76.82), and **ties for first on BigCodeBench (35.80)**" (tie with Claude-4-Sonnet 35.80). Zero-shot, no retraining.

---

## 4. Ablations (Table 2, §4.5, p.6–7) — which components matter

TRINITY (full) vs. each component removed. In-distribution, linear head, 4096 tokens, max 5 turns.

| Method | LiveCodeBench | MATH500 | MMLU | RLPR | Average |
|---|---|---|---|---|---|
| **TRINITY** | **61.46** | **88.00** | 91.56 | **40.72** | **70.44** |
| w/o Singular value fine-tuning | 55.68 | 85.85 | 90.10 | 39.77 | 67.85 |
| w/o Thinker-role selection | 57.80 | 86.20 | **92.75** | 38.00 | 68.69 |
| w/o Tri-role selection | 58.28 | 82.00 | 91.64 | 36.15 | 67.02 |
| w/ Last token (vs penultimate) | 50.85 | 87.00 | 82.19 | 38.60 | 64.66 |
| Claude-4-Sonnet only | 39.09 | 82.25 | 88.23 | 34.90 | 61.12 |
| Gemini Pro 2.5 only | 46.51 | 83.05 | 79.41 | 43.00 | 62.99 |
| GPT-5 only | 59.54 | 75.66 | 90.74 | 37.87 | 65.95 |

Component-importance findings (verbatim, §4.5, p.7):
- Removing SVF "consistently lowers scores."
- Removing tri-role: "substantial degradation on **MATH500 (-6.0 points)** and **RLPR (-4.57 points)**."
- Last token (semantically sparse EOS): "severe performance collapse, particularly on LiveCodeBench (**more than 10 points drop**)" (61.46 → 50.85).
- Ablation (5) "remove agent selection but keep role selection" (single fixed agent): "performance is significantly undermined" (caption notes this case; not given its own numeric row in Table 2).
- Ordering of importance by avg drop: tri-role removal (70.44→67.02, −3.42) ≈ SVF removal (−2.59) ≈ thinker removal (−1.75), last-token swap is worst (−5.78).

---

## 5. Optimizer comparison: sep-CMA-ES vs RL / IL / random (Table 4, §4.8, p.9)

In-distribution, comparable budgets, linear head.

| Method | LiveCodeBench | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| REINFORCE | 0.253 | 0.459 | 0.500 | 0.266 |
| RS (random search, fitness avg) | 0.374 | 0.794 | 0.897 | 0.345 |
| SFT (imitation learning) | 0.592 | 0.786 | 0.906 | 0.360 |
| **sep-CMA-ES** | **0.615** | **0.880** | **0.916** | **0.401** |

RELATIVE claim to reproduce: **sep-CMA-ES > SFT > RS > REINFORCE** on every task (LCB, MATH500, MMLU, RLPR). SFT scores also restated in §A.2.1 (p.18–19): "0.592, 0.786, 0.906, and 0.360 on LiveCodeBench, MATH500, MMLU, and RLPR."

Budget config (§A.5, p.26 / §3.3, p.5): population λ = ⌈4 + 3 ln n⌉ = **32** (n≈10000); m_CMA = **16**, m_RS = **32**; REINFORCE batch = sep-CMA-ES per-iteration eval size, run **60 iterations**; RS warmstarted by sampling uniformly from **[−0.5, 0.5]**, 32 trials per vector. Budget regime: "**1.5k–40k evaluations** for a 10k-dimensional problem" (§1, p.2). Proposition 1 holds for T ∈ [2, 60].

---

## 6. Head-architecture results (Table 3, §4.7, p.8) — "which head"

By varying head (output conversion = softmax by default; block-diagonal-10 uses argmax):

| Head | LiveCodeBench | MATH500 | MMLU | RLPR |
|---|---|---|---|---|
| **linear** (TRINITY default) | **0.615** | **0.880** | 0.916 | **0.401** |
| low-rank | 0.597 | 0.770 | 0.914 | 0.344 |
| sparse | 0.400 | 0.811 | **0.917** | 0.372 |
| block-diagonal-2 | 0.336 | 0.776 | 0.897 | 0.378 |
| block-diagonal-10 + argmax | 0.551 | 0.812 | 0.802 | 0.376 |

Claim: "**linear is the most reliable choice overall**," sparse edges it on MMLU only by a "negligible margin." Block-diagonal-10 "retains a large fraction of the performance despite its tiny parameter count" (1,024 params, 10× fewer than linear's 10,240). Head param sizes (Table 6, p.25): SVF=9216, linear=10240, low-rank=20680, sparse=11266, block-diagonal-2=5120, block-diagonal-10=1024.

---

## 7. Additional baselines (Appendix A.7.3, p.28)

**Majority@5 on MMLU** (Table 7): Gemini Pro 2.5 = **91.57 ± 0.70**; GPT-5 = **91.31 ± 0.23**; Claude-4-Sonnet = **90.99 ± 0.39**. (TRINITY MMLU = 91.56 — competitive without parallel sampling.)

**LLM-as-Coordinator** (Table 8, Gemini 2.5 Pro prompted to pick model+role each turn):
| Method | Math500 | MMLU | RLPR | LiveCodeBench | Avg |
|---|---|---|---|---|---|
| TRINITY | 88.00 | 91.56 | 40.72 | 61.49 | 70.44 |
| Gemini 2.5 Pro as Coordinator | 78.67 | 83.26 | 26.83 | 26.28 | 53.76 |

Claim: "this prompting-based method underperforms TRINITY's trained coordinator (**64.14 vs 70.44** average score)" — note the text says 64.14 but Table 8 gives **53.76**; see OPEN section for the discrepancy.

---

## 8. Token-efficiency results (Appendix A.7.4, p.29–30) — fairness of comparison

**Table 9 — Avg output tokens, coordination methods:**
| Model | Math500 | MMLU | RLPR | LiveCodeBench |
|---|---|---|---|---|
| TRINITY | 2,853 | 1,200 | 2,141 | 1,999 |
| MoA | 6,871 | 5,218 | 11,086 | 21,634 |
| RouterDC | 624 | 374 | 811 | 1,552 |
| Smoothie | 6,472 | 4,718 | 10,580 | 17,864 |
| MasRouter | 4,260 | 1,847 | 5,370 | 8,401 |

Claim (§4.2, p.7): "TRINITY exhibits upper-tier token efficiency... especially [vs.] coordination methods." TRINITY uses far fewer tokens than MoA/Smoothie/MasRouter while scoring higher. Tables 10–12 give per-model token usage under 5×SR, 5×CTX, and default-4096 contexts (used to define the "fair, adjusted token budget" single-model comparison: 4K and 20K=5x).

---

## 9. Representation/objective-separability supporting metrics (§4.6, §A.3, p.8, 22–23)

- Linear SVM task-type classification on penultimate hidden states: **Accuracy 1.000 ± 0.000** (random 0.250, 1698 samples). RBF SVM task-type = **1.000**.
- On head logits (Table/Fig.13): Linear SVM task-type = 0.945, RBF = 0.955; agent selection linear = 0.786, RBF = 0.783.
- On SLM hidden states (Fig.12): agent selection linear SVM = 0.713 ± 0.016, RBF = 0.776 ± 0.020.
- Agent-selection SVM accuracy per task correlates with TRINITY's relative advantage: LiveCodeBench **0.844**, MATH500 **0.764**, MMLU **0.679**, RLPR **0.544** (§A.3, p.23).
- LDA Fisher's ratios ≈ 2.85 (from SLM) and 2.45 (from head).

---

## Target-metrics table we will aim to reproduce (our pool: deepseek-v4-pro, glm-5p2, kimi-k2p6)

Absolute numbers WILL differ (different 3-model pool, no GPT-5/Gemini/Claude/open-source 32B models). Capture these RELATIVE invariants:

| # | Claim to reproduce | Paper evidence |
|---|---|---|
| R1 | TRINITY avg > best single model avg (5x-budget-matched) | 70.44 > GPT-5 65.95 (Table 2); 54.21 > Gemini 52.34 (Table 1) |
| R2 | TRINITY > every single model on every in-dist + held-out task | Tables 1, 2 |
| R3 | TRINITY > best multi-agent baseline (MoA/MasRouter/RouterDC/Smoothie) on every task | Fig.3, §4.2 |
| R4 | TRINITY > random routing | RLPR random 0.32; TRINITY 0.41 (Fig.3) |
| R5 | TRINITY ≈ Per-Question-Best on 3 of 4 in-dist tasks | §4.2, §4.3 |
| R6 | Lifting token cap → large LCB jump approaching SOTA, beats all constituents | 0.61 → 0.862; beats GPT-5 0.838 (§4.4) |
| R7 | More max-turns → monotonic gain | 0.823→0.863 (2→6 turns) |
| R8 | sep-CMA-ES > SFT > RS > REINFORCE on all 4 tasks | Table 4 |
| R9 | Removing SVF / thinker / tri-role / penultimate-token all hurt; tri-role + token choice matter most | Table 2 |
| R10 | linear head ≥ all other heads overall | Table 3 |
| R11 | Trained coordinator > LLM-as-coordinator | Table 8 |
| R12 | TRINITY far more token-efficient than MoA/Smoothie/MasRouter | Table 9 |
| R13 | Mean relative error reduction ≈ 21.9% vs 2nd-best (target ballpark, pool-dependent) | §1 |

Key fixed hyperparameters for our run (verbatim, §4.1, p.5; §A.5; §A.2): coordinator SLM = **Qwen3-0.6B**; head = **single linear layer, 10K params** (10,240); SVF on **second-to-last layer** (9,216 params); total learnable **< 20K**; max generated tokens **4096** default with **minimal reasoning effort**; **max 5 coordination turns**; pool size in paper = **7**; sep-CMA-ES λ=**32**, m_CMA=**16**, m_RS=**32**; SFT used Adam, lr **1×10⁻⁶**, batch **64**; low-rank head bottleneck r=**14**, ELU α=**0.1**; sparse-head Gumbel temperature τ ∈ [1.0, 20.0]. LiveCodeBench: V1 (**400 samples**) train, V6 (**175 samples**) eval. Reward R(τ) ∈ {0,1} terminal.

---

## OPEN / AMBIGUOUS (details absent or inconsistent in the paper)

1. **Fig.3 baseline cells are not in a machine-readable table.** Only TRINITY rows (Tables 2–4), single-model-only rows (Table 2), and a few text-cited baseline points (MoA: Math500 0.83 / RLPR 0.38 / LCB 0.39; RouterDC RLPR 0.28; random RLPR 0.32) are unambiguous. The exact per-task scores for MasRouter, Smoothie, MoA-MMLU, RouterDC-non-RLPR, self-reflection, 5x-CTX, majority-vote bars, etc., must be read off the bar chart OCR (lines 415–454), which is partially garbled and unreliable. We cannot quote most individual baseline cells with confidence.

2. **LLM-as-Coordinator average discrepancy.** §A.7.3 text says "64.14 vs 70.44" but Table 8 reports the Gemini-coordinator average as **53.76** (Math500 78.67, MMLU 83.26, RLPR 26.83, LCB 26.28). The 64.14 figure is not reconstructable from the table; treat as a paper inconsistency.

3. **TRINITY LiveCodeBench value rounding inconsistency:** reported as 61.46 (Table 2), 0.615 (Table 3 / Table 4), 0.61 (§4.2 text), and 61.49 (Table 8). Likely the same run; we should target ≈61.5.

4. **Per-Question-Best numeric values** are only shown as Fig.3 bars (≈0.91 Math500, ≈0.97 MMLU, ≈0.65 LCB; RLPR not clearly legible). No table.

5. **Number/identity of seeds for reported means** is only partially specified (SFT label gen used "3 seeds on 7k datapoints"; main-table seeds and the ±0.5% LCB error-bar replication count are not explicitly stated for every cell).

6. **MT-Bench scale** (Table 1) is a ~10-point judge score, not a percentage; the "Average" column mixes percentage tasks with this 10-point score, so the averaging convention is unstated (cannot verify how 9.60 is folded into 54.21). Treat MT-Bench separately when reproducing.

7. **Exact decoding params (temperature, top-p) for the pooled LLMs** are NOT given anywhere — only "maximum generated tokens 4096" and "minimal reasoning effort." Temperature is unspecified for every model and turn. This is a needed implementation detail that is absent.

8. **Role-specific prompt templates** (Thinker / Worker / Verifier injected prompts) are described functionally (§3.2, p.4) and illustrated by example (Fig.1, p.1), but no verbatim prompt template strings are provided. The ACCEPT/REVISE verifier output contract is described but its exact prompt is not given.

9. **Pool size for our replication = 3, not 7.** Several paper results depend on a 7-model pool (e.g., the LCB "narrows to the three closed models after training" §4.4; agent-distribution Fig.15). With only 3 models, the agent-selection-distribution claims (Fig.6, Fig.15) and the "narrowing to closed models" mechanism may not transfer; absolute Per-Question-Best ceilings differ. Capture relative claims (R1–R13) only.

---

## EXTRACTION: APPENDIX

I have now read the entire paper (all 2532 lines). I have all the implementation and reproducibility details from the appendix, footnotes, main text, and tables. Here is my dense Markdown report focused on my dimension (APPENDIX / IMPLEMENTATION & REPRODUCIBILITY).

---

# TRINITY — APPENDIX / IMPLEMENTATION & REPRODUCIBILITY DIMENSION

Source: `/home/cybernovas/Desktop/2026/experiments/trinity/docs/paper/trinity_paper.txt` (ICLR 2026, arXiv:2512.04695v3). All numbers below are quoted verbatim with section/page citations. Where a value is needed to implement but not given, it is flagged under **OPEN / AMBIGUOUS**.

---

## 1. MASTER HYPERPARAMETER / DEFAULT-VALUE TABLE

Every concrete default I could find, with citation. (Note the replication research uses Fireworks-served deepseek-v4-pro / glm-5p2 / kimi-k2p6; the *paper's* pool is different — listed verbatim below for fidelity.)

| Parameter | Value | Source |
|---|---|---|
| Coordinator SLM | **Qwen3-0.6B** | §4.1 (p.5) |
| SLM hidden-state dim `d` = `dh` | **1024** | §A.3 (p.23) "1024 dimensions"; §A.4 examples "dh = 1024" (p.24-25) |
| Default head | **linear**, single linear layer, **10K params** (`dh·na = 10,240`) | §4.1 (p.5); Table 6 (p.25) |
| Head output size | **L + 3** = `na` logits (L agent logits + 3 role logits) | §3.1 (p.3) |
| `na` (total logits) | **10** (L=7 agents + 3 roles) | §A.4 (p.25) "na = 10"; Fig.2 |
| SVF target layer | **second-to-last layer** of the 0.6B model | §4.1 (p.5) |
| SVF param count | **9216** | Table 6 (p.25) |
| Total learnable params | **< 20K** (head + SVF; e.g. 10240 + 9216 = 19,456) | Abstract; §1; §3.1; Table 6 |
| Max coordination turns `K` = `Bturn` | **5** | §4.1 (p.5); §4.4 "maximum turn setting of five" |
| Max generated tokens per LLM (default) | **4096** | §4.1 (p.5); Table 12 caption "Default Context (4096)" |
| Reasoning effort (default) | **minimal** ("minimal reasoning effort") | §4.1 (p.5) |
| Hidden state used | **penultimate output token** (NOT last/EOS) | §2; Fig.2; §4.5 ablation |
| Output conversion (default) | **softmax** normalization (argmax only for block-diagonal-10) | Table 3 caption (p.8) |
| Optimizer | **sep-CMA-ES** (separable CMA-ES, diagonal covariance) | §3.3; §A.1 |
| Head dimension `n` (CMA problem dim) | **≈ 10000** ("n ≈ 10000", 10k-dimensional) | §1; §A.1.2 (p.17) |
| CMA-ES population size `λ` | **⌈4 + 3 ln n⌉ = ⌈4 + 3 ln 10000⌉ = 32** | §3.3 (p.5); §A.1 (p.17) |
| CMA replication count `mCMA` | **16** | §3.3; §A.1.2 (p.17) |
| RS replication count `mRS` | **32** | §3.3; §A.1.1 (p.15); §A.5 |
| Total optimization iterations `T` | small-T regime, **T ∈ [2,60]** (Prop.1); REINFORCE & implied CMA ran **60 iterations** | Prop.1 (p.18); §A.5 (p.27) |
| Total atomic eval budget `Benv` | **1.5k–40k evaluations** for the 10k-dim problem | §1 (p.2) |
| Reward `R(τ)` | binary terminal, **∈ {0,1}** (Bernoulli) | §2 (p.3) |
| RS warmstart sampling range | **U[−0.5, 0.5]** (band slightly exceeding sep-CMA-ES weight extrema) | §A.5 (p.27) |
| RS trials per sampled vector | **32** | §A.5 (p.27) |
| REINFORCE batch size | = per-iteration eval size of sep-CMA-ES; ran **60 iterations** | §A.5 (p.27) |
| SFT optimizer | **Adam** (frozen SLM, train only linear head) | §A.2.1 (p.19) |
| SFT learning rate | **1 × 10⁻⁶** (best after sweep) | §A.2.1 (p.19) |
| SFT batch size | **64** (best after sweep) | §A.2.1 (p.19) |
| Low-rank head bottleneck `r` | **14** | §A.4 (p.24) |
| Low-rank head ELU `α` | **0.1** | §A.4 eq.6 (p.24) |
| Low-rank head scale `σ` | fixed, non-trainable scalar (value unspecified) | §A.4 eq.7 (p.24) |
| Low-rank head init | **Xavier-uniform** with adaptive gains (eq.8) | §A.4 (p.24-25) |
| Sparse head Gumbel temperature `τ` | **∈ [1.0, 20.0]** (annealed) | §A.4 eq.10 (p.25) |
| Sparse head active-dim target `k` | `max(1, ⌊dh·(1−sigmoid(ρ))⌋)`, ρ learnable | §A.4 (p.25) |
| Candidate-filter quantile | **top 5%** (τ = 95th percentile of E(Dy,Mx)) | §A.6 eq.16 (p.26) |
| Candidate pool size | **7 models** (3 closed + 4 open) | §4.1 (p.5) |

---

## 2. HEAD ARCHITECTURE — FULL SPEC (§A.4, p.24–25)

Maps `h ∈ R^{dh}` → logits `z ∈ R^{na}`, then softmax (default) or argmax.

**Linear head (DEFAULT):** `z = Wh, W ∈ R^{na×dh}`, **no bias**. Exactly `dh·na` params (10,240). Eq.5.

**Low-rank head:** `u = ELU(Uh)`, ELU with α=0.1; `z = Vu·σ`; `U ∈ R^{r×dh}`, `V ∈ R^{na×r}`, fixed non-trainable scalar σ. `r = 14`. Init (eq.8, Xavier-uniform adaptive gains):
`U ∼ U[−√(6/(dh+r)), √(6/(dh+r))]`, `V ∼ U[−√(18/(r+na)), √(18/(r+na))]`. Param size 20,680.

**Sparse head:** `z = W(h ⊙ α)`, `W ∈ R^{na×dh}`; α is a data-agnostic learnable selection vector. `k = max(1, ⌊dh·(1−sigmoid(ρ))⌋)`, ρ learnable. Training uses differentiable Gumbel top-k: `s̃ = (s+ε)/τ`, ε∼Gumbel(0,1), τ∈[1.0,20.0]; `αsoft = TopKsoft(s̃,k)`, `α = αsoft·k / Σ αsoft,i`. Inference uses hard top-k binary mask. Params = `dh·na + dh + 2` (= 11,266).

**Block-diagonal head:** block-diagonal W (eq.12). Two variants:
- **block-diagonal-2** (B=2): partitions hidden + agent/role dims proportionally. Params 5,120.
- **block-diagonal-10** (B=10, one block per logit, ai=1): hidden dims distributed evenly, `hj = ⌊dh/10⌋+1` for `j ≤ (dh mod 10)` else `⌊dh/10⌋`. Uses **argmax** output conversion. Params 1,024 (exact 10× reduction vs linear).

**Table 6 (p.25) — Parameter sizes (Qwen3-0.6B, dh=1024):**

| SVF | linear | low-rank | sparse | block-diag-2 | block-diag-10 |
|---|---|---|---|---|---|
| 9216 | 10240 | 20680 | 11266 | 5120 | 1024 |

---

## 3. PSEUDOCODE / ALGORITHM STRUCTURE

The paper gives **no formal numbered algorithm/pseudocode listing**. The procedure is described prose-only:

**sep-CMA-ES loop (§3.3, p.4):** "iteratively improves a central 'parent' policy by sampling a population of perturbed parameter vectors, evaluating each candidate to obtain a fitness score, and recombining candidates via fitness-weighted averaging to form the next parent." Maintains only diagonal covariance.

**CMA mechanics (§A.1.1, p.15):** mean `mt ∈ P`, step-size `σt > 0`, diagonal scaling `Dt = diag(√s1,t,…,√sn,t) ≻ 0`; sampling `y = mt + σt Dt z, z ∼ N(0, In)`; whitened chart `x = Dt⁻¹(y−mt) = σt z`. `μ` parents with weights `(wj)_{j=1}^μ`; `zj:λ` are order statistics. Diagonal learning rate `ccov = Θ(1/n)`.

**Multi-turn coordination (§3.2, p.4):** transcript `Ck−1 = (Q, O1,…,Ok−1)`. At turn k: coordinator selects agent `Ak` from pool M and role `Rk ∈ {T,W,V}`; prepares role-specific prompt from `Ck−1`; queries `Ak` → message `Mk`; post-processes `Mk` → `Ok`; appends to transcript. Termination: `τ = min{k ≤ K : Rk = V and uk = ACCEPT}`, else `τ = K`. Final answer = `Oτ`. Verifier outputs `uk ∈ {ACCEPT, REVISE}` + optional diagnosis `δk`; coordinator records `(uk, δk)`.

---

## 4. PROMPT TEMPLATES

**No verbatim role prompt templates are given.** Only role *contracts* and one illustrative worked example (Fig.1, p.2) are provided. Implementer must author the actual Thinker/Worker/Verifier prompts.

Role contracts (§3.2, p.4):
- **Thinker** — "returns meta-level guidance, including high-level plans, decompositions, or critiques of partial solutions… may propose a plan over subgoals… can also specify the role of the next agent along with the plan."
- **Worker** — "acts directly on the task… produces actionable content (e.g., a derivation, code snippet, or numerical result)."
- **Verifier** — "checks whether the accumulated solution… is correct, complete, and responsive to Q. Outputs a judgment uk ∈ {ACCEPT, REVISE} and an optional diagnosis δk."

Fig.1 example messages (verbatim, depreciation problem):
- Msg processing/Thinker (Turn 1): "The user requires the calculation of the second-year depreciation expense for a vehicle using the double declining-balance method. The initial response is empty. The next agent should act as a solver to perform the calculation." then a numbered step-by-step plan.
- Verifier (Turn 3): "The calculation provided in the current response appears to be correct and follows the double declining-balance depreciation method accurately… However, one potential issue to verify is whether the double declining-balance method should switch to straight-line depreciation if it results in a higher depreciation amount in later years…"

Note (Fig.1): "A message processing module injects a role-specific prompt before the request is sent to the chosen LLM." Coordinator-generated text is **discarded**; only head logits are used (§3.1, p.4).

---

## 5. CONTEXT / TOKEN / TOKENIZER LIMITS

- Max output tokens per LLM: **4096** default; **20480 (5× CTX)** for budget-matched baselines (§4.1, §A.7.1). The 5× = matches 5-turn setting.
- Coordinator input: "concatenating the original query with the full transcript of prior turns" (§1). Each LLM input = role-specific prompt over `Ck−1` (§3.2). **No explicit coordinator context-length cap stated** beyond Qwen3-0.6B's native limit.
- Coordinator can use **an earlier token's hidden state instead of penultimate** for faster inference ("this allows the coordinator to take hidden states corresponding to an earlier token instead of the penultimate to make a quick decision", §3.1, p.4) — though all experiments/ablations use penultimate.
- **Tokenizer:** not specified beyond "Qwen3-0.6B." No special-token handling detailed except `<\think>` / EOS referenced as the penultimate-token's predicted next token (§1, p.1).

**Average output token usage — TRINITY (Table 9, p.29):** Math500 2,853 / MMLU 1,200 / RLPR 2,141 / LiveCodeBench 1,999. (Comparison: MoA 6,871/5,218/11,086/21,634; RouterDC 624/374/811/1,552; Smoothie 6,472/4,718/10,580/17,864; MASRouter 4,260/1,847/5,370/8,401.) Per-model token tables for 5×SR, 5×CTX, and Default(4096): Tables 10–12 (p.29–30).

---

## 6. FITNESS AGGREGATION / EVALUATION PROTOCOL

- **Atomic evaluation** = one complete end-to-end trajectory τ; reward is a **Bernoulli call** (binary, terminal) (§2, p.3).
- `J(θ) = E_{τ∼πθ}[R(τ)]`; estimated "via replication/averaging" — `Benv` counts individual Bernoulli calls (§2).
- Fitness per candidate = **average reward over m replications** (m=16 CMA / m=32 RS). RS explicitly: "run 32 independent trials… compare the average reward" (§A.5).
- **Cross-task aggregation during training:** Trained **per-task** ("For each task, we train on the designated training set and assess performance on the corresponding test set," §4.1). No single multi-task fitness blend is described — each benchmark gets its own coordinator/run. The Average columns in Tables are reporting means, not training objectives.
- LiveCodeBench split: **train on V1 (400 samples)**, eval on **V6 newly-introduced questions (175 samples)**, Jan–Apr 2025 (§4.1).
- Held-out (zero-shot, no retraining): AIME2025, BigCodeBench, MT-Bench, GPQA-D (§4.1).
- "Unleashing full power" (§4.4): for LiveCodeBench, output-length constraint **removed without retraining**; selection had already narrowed to the 3 closed models → 86.2% pass@1.

---

## 7. CACHING / PARALLELIZATION / WALL-CLOCK / COST

- **LLM response caching:** **NOT mentioned anywhere.** (OPEN)
- **Parallelization:** Not explicitly described, though sep-CMA-ES population (λ=32) × replication (16) candidates are naturally parallelizable. No infra detail beyond acknowledgement: "We thank Koshi Eguchi and Kou Misaki for the infrastructure support" (Acknowledgements, p.11).
- **Wall-clock:** **Not reported.** (OPEN)
- **Cost (the one concrete cost analysis, §A.2.2, p.19):** SFT label generation cost only.
  - Direct single-step labeling: `3 seeds × 7k datapoints × 7 agents = 147k = 1.5×10⁵ LLM queries`.
  - Multi-turn label complexity: factor `7⁴ ≈ 2.4×10³` (agent selection over 4 remaining turns) × `3⁵ = 243 ≈ 2.4×10²` (role selection over 5 turns) = `7⁴·3⁵ = 583,443 ≈ 5.8×10⁵`.
  - Total multi-turn SFT cost: `1.5×10⁵ × 5.8×10⁵ ≈ 8.7×10¹⁰ LLM queries` (cited as why SFT is infeasible multi-turn).
  - sep-CMA-ES / REINFORCE / RS: "require no explicit label generation."
- Hardware note: "Due to hardware constraints in serving open-source models, we limited the maximum output length for all LLMs" (§4.4) — implies local serving of the 4 open models; no GPU type/count given.

---

## 8. RANDOM SEEDS

- SFT label gen uses **3 seeds** ("running 3 seeds on 7k datapoints," §A.2.2; "for each seed independently… aggregate across all seeds via majority voting," §A.2.1).
- Reported scores carry ± std (e.g., LiveCodeBench pass@1 **86.2 ± 0.5%**, §1; Fig.4 values 0.862±0.005, 0.838±0.007, 0.672±0.031, 0.465±0.010; Table 7 MMLU majority@5 with ±; SVM accuracies "1.000 ± 0.000", "0.713 ± 0.016", etc.) implying multi-seed eval, but **exact seed values and number of eval seeds are not stated.** (OPEN)

---

## 9. HEAD INITIALIZATION

- **Linear head (default):** **NO initialization scheme stated.** Only low-rank head init is specified (Xavier-uniform adaptive gains, eq.8). For SVF, only "learn the singular value scales, keeping orthogonal matrices fixed" — initial scale values unspecified. (OPEN, partly)
- sep-CMA-ES initial mean `m0`: re-centering "w.l.o.g." to origin in analysis (§A.1.1); actual init mean of θ for training **not given**. Initial `σ0` **not given**. (OPEN)

---

## 10. SVF (SINGULAR VALUE FINE-TUNING) DETAILS (§3.1, p.3; §4.1)

- Method from "Transformer-squared / Sun et al. 2025" (Transformer²): SVD a selected subset of weight matrices, **learn only singular-value scales**, keep U,V (orthogonal matrices) fixed.
- Applied to **second-to-last layer** of Qwen3-0.6B.
- 9216 trainable params (Table 6).
- Which specific weight matrices within that layer (Q/K/V/O/MLP) are decomposed: **not specified.** (OPEN)

---

## 11. SEPARABILITY / DIAGNOSTIC EXPERIMENT SETTINGS (§A.3, p.20–24)

- SVM probe: features standardized, class-balance controlled; **Samples: 1698** (Figs 12–13).
- Synthetic separability study: **1024 dims, 7 agent classes, 4 task-type clusters**; control via scaling inter-class-center distance with consistent within-class covariance; separability index = between-class variance / total variance (§A.3, p.23).
- Linear SVM agent-selection accuracy by task (from SLM hidden states): LiveCodeBench 0.844, MATH500 0.764, MMLU 0.679, RLPR 0.544 (§A.3, p.23).
- LDA Fisher ratios ≈ 2.45–2.85 (Fig.9).

---

## 12. DATASET–AGENT SUBSET SELECTION (§A.6, p.25–27)

- Joint subset selection maximizing **Relative Error Reduction** `RER(C,M') = (Z_{C,M'} − S*_{C,M'}) / (1 − S*_{C,M'})` (eq.14, 25–27).
- `Z` = best-per-dataset accuracy (coordinated upper proxy); `S*` = best single-agent baseline on the subset.
- Procedure: build full performance matrix `E(Dy,Mx)`; **top-5% frontier filter** (τ=95th pct, eq.16); **exhaustive enumeration** of all (C,M') under coverage constraints `|C|≤Y, |M'|≤X`; tie-break by task/agent diversity (favor mix of reasoning + direct-inference agents). Fallback heuristics (greedy + annealed/beam) suggested for larger spaces.

---

## 13. PAPER'S MODEL POOL (for fidelity — replication uses different Fireworks models)

7 agents (§4.1; Fig.15 IDs, p.28): **A0 GPT-5, A1 Claude-Sonnet-4-20250514, A2 Gemini-2.5-pro, A3 DeepSeek-R1-Distill-Qwen-32B, A4 Gemma-3-27B-It, A5 Qwen3-32B (reasoning), A6 Qwen/Qwen3-32B (direct)**. Note Fig.12/13 reference "Agent 0,1,2,6" (the 4 most-selected). Per-task selection % (Fig.15): LCB → A0 87.9, A1 9.3, A2 2.3, A6 0.5; Math500 → A0 9.2, A1 5.3, A2 82.2, A6 3.4; MMLU → A0 24.7, A1 74.3, A2 1.0; RLPR → A0 22.2, A1 16.3, A2 59.4, A6 2.1. Open models (A3–A5) ~0% selected post-training.

---

## OPEN / AMBIGUOUS (details absent or under-specified for faithful re-implementation)

1. **Exact role prompt templates** (Thinker/Worker/Verifier) — only contracts + one illustrative example given. Must be authored.
2. **Message post-processing module** ("lightly post-processes Mk into Ok," "coordinator condenses/extracts") — algorithm/prompt unspecified.
3. **LLM response caching** — not mentioned at all.
4. **Wall-clock time and total dollar/compute cost** of a TRINITY training run — not reported (only SFT label-gen cost arithmetic).
5. **Parallelization scheme / GPU type / count** — only an acknowledgement of infra support; no spec.
6. **Random seed values** and **number of eval seeds** for reported ± — not stated (SFT label-gen uses 3 seeds).
7. **Linear head initialization** — unspecified (only low-rank head init given via eq.8).
8. **sep-CMA-ES init**: initial mean `m0`, initial step-size `σ0`, μ value, weight scheme `wj`, learning rates beyond `ccov = Θ(1/n)` — not given numerically (μ and wj only appear symbolically).
9. **SVF specifics**: which weight matrices (attention vs MLP) in the second-to-last layer are SV-fine-tuned; initial singular-value scale values.
10. **Fixed scale σ** in the low-rank head (eq.7) — declared "fixed non-trainable" but value not given.
11. **Cross-task fitness blending** — training appears strictly per-task; no multi-task aggregate objective documented (the H200/single-GPU replication must decide whether to train one coordinator per benchmark, matching the paper, or a shared one).
12. **Exact `Benv` per task** — only the range "1.5k–40k evaluations" is given, not per-benchmark budgets.
13. **`Y` and `X` coverage-constraint values** in subset selection (eq.25) — unspecified (final result: 7 agents, 4 datasets).
14. **REINFORCE hyperparameters** (LR, baseline, entropy) — only "batch size = sep-CMA-ES per-iteration eval size, 60 iterations."
15. **Verifier ACCEPT/REVISE parsing** — how `uk` is extracted from free-text verifier output is unspecified.
16. **Reasoning-effort knob** ("minimal reasoning effort") — provider-specific; mapping to Fireworks models undefined.
17. **`d` vs `dh`**: paper states `dh = 1024`; Qwen3-0.6B's actual hidden size should be verified against this when implementing (paper asserts 1024).