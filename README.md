# TinyRouter

We built a small **coordinator** that, for every question, decides two things: **which** of three
open-source LLMs should answer it, and **what role** that model should play (Thinker, Worker, or
Verifier). The coordinator is deliberately tiny and cheap. A frozen **0.6B** encoder reads the
question into a single vector, and a **~10K-parameter** head turns that vector into the routing
decision. It is trained by **separable CMA-ES**, a derivative-free evolution strategy, against a simple
right/wrong reward. The coordinator never solves the question itself; it only learns who to ask.

The method follows TRINITY (Xu et al., ICLR 2026, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
rebuilt from scratch with an all open-source model pool served through Fireworks AI.

## What we did

- Implemented the full coordinator: the 0.6B encoder feature, the ~10K routing head, the three roles,
  the multi-turn loop (up to 5 turns, terminated by a Verifier accept), and the sep-CMA-ES trainer.
- Wired a 3-model open-source pool plus an automatic grader (exact-match for math, letter-match for
  MMLU) that produces the binary reward.
- Trained per-task coordinators by evolution: breed thousands of candidate heads, keep the ones that
  route best, repeat.
- Evaluated rigorously on 120 held-out questions, with every single-model baseline averaged over 3 runs
  to remove run-to-run noise, against each model alone and against random routing.
- Built an **oracle-ceiling diagnostic** to ask whether the pool even leaves room for routing to help,
  and used it to decide where improvement effort was worth spending.
- Implemented and tested two upgrades from that diagnostic (supervised warm-start of the head, shaped
  training fitness) and measured them on the task with real headroom.
- Tracked every dollar of API spend and logged each result.

## Model pool

| Slot | Model            | Strong at        |
| ---- | ---------------- | ---------------- |
| A    | `deepseek-v4-pro` | knowledge (MMLU) |
| B    | `glm-5p2`         | math             |
| C    | `kimi-k2p6`       | general          |

The 0.6B encoder and the evolution loop run on a single NVIDIA H200; the three LLMs are called over HTTP.

## How it works

1. The frozen 0.6B encoder turns the question into one 1024-dim vector.
2. The ~10K head reads that vector and picks a model and a role.
3. The chosen model answers in that role; its output is appended to the transcript.
4. Steps 1 to 3 repeat for up to 5 turns; a Verifier turn can accept and stop early.
5. The final answer is graded right/wrong, and that reward drives the evolutionary training.

## Results

Rigorous eval: 120 held-out questions per task; single-model baselines are the mean over 3 runs.
Scores are fraction correct (0.792 = 79.2%).

**Math**

| system | score |
| --- | --- |
| glm-5p2 | 0.794 (best single) |
| **TinyRouter** | **0.792** |
| random routing | 0.792 |
| deepseek-v4-pro | 0.747 |
| kimi-k2p6 | 0.742 |

**MMLU**

| system | score |
| --- | --- |
| **TinyRouter** | **0.925** |
| deepseek-v4-pro | 0.922 (best single) |
| random routing | 0.875 |
| glm-5p2 | 0.783 |
| kimi-k2p6 | 0.539 |

**Both tasks together**

| system | math | MMLU | average |
| --- | --- | --- | --- |
| **TinyRouter** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 | 0.922 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | 0.794 | 0.783 | 0.789 |
| kimi-k2p6 | 0.742 | 0.539 | 0.640 |

### What the numbers say

The tiny router scores **0.858 on average, higher than any single model**. No single model is good at
both tasks: deepseek is the knowledge specialist, glm is the math specialist. The router wins the
average by sending each task to the right specialist.

Reading it straight: the win is **across tasks, not within a task**. On MMLU, where the models differ a
lot (0.54 to 0.92), routing clearly helps and the router beats random (0.925 vs 0.875). On math, where
all three models sit around 0.79, there is nothing to route around, so the router ties both the best
model and random routing. Routing pays off when the models genuinely differ.


| benchmark | best single | perfect router | real headroom (95% CI) | verdict |
| --- | --- | --- | --- | --- |
| math500 | 0.808 | **0.856** | **+0.049 [0.005, 0.085]** | ROUTER_BOUND |
| MMLU | 0.939 | ≥0.939 | +0.025 [0.000, 0.058] | inconclusive (near-ceiling) |

This overturned the easy reading of math as "no benefit." There **is** about 4.9 points of real,
achievable headroom on math; our trained router just captures none of it. So the math limit is the
router, not the pool. MMLU sits near its ceiling, where deepseek already dominates and the router
already matches it.

## Trying to capture it: warm-start + shaped fitness

The diagnostic pointed effort at math, so we tried two upgrades: warm-starting the head with a
supervised fit against per-(question, model) correctness labels (instead of starting the evolution from
a blank head), and shaping the training reward (format bonus, turn penalty, variance reweighting) while
keeping the **eval pure right/wrong**.

| system | math (held-out 120) |
| --- | --- |
| best single (glm-5p2) | 0.817 |
| **TinyRouter (warm-start + shaped)** | **0.808** |
| prior router, same test | 0.792 |
| random routing | 0.733 |

The eval samples each model once per question, and that sampling noise is large: random routing alone
swung from 0.792 to 0.733 between runs with nothing changed. A swing that size swamps a 1.6-point router
delta. We did not run the clean control (blank-init, pure-binary, same settings), so there is no causal
claim that warm-start or shaping moved the number. The result is still below the best single model
(0.817) and below the 0.856 ceiling, so the headroom the diagnostic found remains on the table. The two
upgrades are implemented and covered by 54 offline tests; whether they move the held-out score is
unproven.

## Cost

Tracked exactly from the token ledgers at real Fireworks prices:

- Core replication and rigorous eval: **$20.89** (deepseek $6.56, glm $6.70, kimi $7.64).
- Oracle-ceiling diagnostic: **~$14**.
- Warm-start + shaped-fitness experiment (label collection, retrain, eval): **$27.22**.
