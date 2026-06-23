# TRINITY Replication — Results

> Structured scorecard for the open-source replication of **TRINITY: An Evolved LLM Coordinator**
> (Xu et al., ICLR 2026) using `deepseek-v4-pro`, `glm-5p2`, `kimi-k2p6` via Fireworks.
>
> The definitive numbers are the **rigorous n=120 evals** in §1–§2 (raw data:
> `experiments/final/math_rigorous.json`, `experiments/final/mmlu_rigorous.json`). The earlier n=40
> per-coordinator runs (§3) are kept for history but are **superseded** — they were too noisy to trust.
> Method: [`AGENTS.md`](../AGENTS.md) · spec: [`SPEC.md`](SPEC.md) · full lab log incl. every
> mistake/fix: [`JOURNAL.md`](JOURNAL.md).

## 0. TL;DR (honest)

On our 3-model open-source pool, the paper's core claim **reproduces on the multi-task average, but
thinly, and the win is cross-task rather than within-task.**

- **Multi-task average — R1/R2 ✅, R4 ✅ (thin):** TRINITY **0.858** beats every fixed single model
  (best is deepseek at **0.835**) and random routing (**0.833**). Margin ~0.02.
- **Per-task MMLU — routing helps:** TRINITY **0.925** clearly beats random (**0.875**) and edges the
  best single model (deepseek 0.922).
- **Per-task math — routing does NOT help:** all three models cluster at ~0.79, so there is no headroom.
  TRINITY (**0.792**) ties the best single model (glm **0.794**) and ties random routing (**0.792**)
  *exactly*. Both invariants are **false** for math as a standalone task.

The mechanism: no single fixed model is good at both tasks (deepseek = knowledge specialist, glm = math
specialist). TRINITY picks the right specialist per task, so its **average** beats any fixed choice. But
**within** a benchmark where the models are equally good, a learned router has nothing to exploit and
matches random. Routing helps across *heterogeneous* tasks, not among *similar* models.

## 1. Rigorous multi-task summary (the paper's R1/R2/R4) — n=120, baselines averaged ×3 reps

| system | math500 | MMLU | **average** |
|---|---|---|---|
| **TRINITY** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro (fixed) | 0.747 | 0.922 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 (fixed) | **0.794** | 0.783 | 0.789 |
| kimi-k2p6 (fixed) | 0.742 | 0.539 | 0.640 |

- **R1/R2 ✅ HOLDS (thin):** TRINITY avg 0.858 > best fixed single avg 0.835 (deepseek). Margin +0.024.
- **R4 ✅ HOLDS (thin):** TRINITY avg 0.858 > random avg 0.833. Margin +0.025.

## 2. Rigorous per-benchmark detail (n=120; single-model baselines = mean ± std over 3 reps)

### math500 — routing gives no benefit (models too similar)

| system | score | note |
|---|---|---|
| glm-5p2 | 0.794 ± 0.017 | best single |
| **TRINITY** | **0.792** | ties best single & random |
| deepseek-v4-pro | 0.747 ± 0.014 | |
| kimi-k2p6 | 0.742 ± 0.018 | |
| random routing | 0.792 | identical to TRINITY |

R1/R2 ❌ (0.792 vs 0.794, inside noise) · R4 ❌ (0.792 = 0.792, exact tie). All three models within
~0.05 of each other ⇒ no complementarity to route around.

### MMLU — routing helps

| system | score | note |
|---|---|---|
| **TRINITY** | **0.925** | beats random, edges best single |
| deepseek-v4-pro | 0.922 ± 0.010 | best single |
| random routing | 0.875 | |
| glm-5p2 | 0.783 ± 0.007 | |
| kimi-k2p6 | 0.539 ± 0.004 | |

R1/R2 ≈ (0.925 vs 0.922, edge within noise) · R4 ✅ (0.925 > 0.875). Models are spread (0.54–0.92) ⇒
real routing headroom, and TRINITY captures it.

## 3. Earlier n=40 per-coordinator evals (SUPERSEDED — kept for history)

These point estimates were unreliable: reasoning models are not fully deterministic, so the *same*
baseline swung 0.45–0.79 across runs at n=40. The n=120 rigorous numbers above replace them. Notably,
the n=40 math story (TRINITY 0.55 > glm 0.50) looked like a routing win but was **small-sample noise** —
at n=120 it is a tie.

| benchmark | coordinator | TRINITY | best single (model) | random |
|---|---|---|---|---|
| math500 | full_pilot | 0.550 | 0.500 (glm-5p2) | 0.325 |
| math500 | math_s1 | 0.525 | 0.450 (glm-5p2) | 0.400 |
| math500 | math_s0 | 0.325 | 0.700 (glm-5p2) | 0.425 |
| mmlu | mmlu_s1 | 0.950 | 0.975 (deepseek) | 0.850 |
| mmlu | mmlu_s0 | 0.925 | 0.950 (deepseek) | 0.875 |

## 4. Honest caveats (do not over-read)

1. **The multi-task win is thin (~0.02) and cross-task.** It comes from picking the right specialist per
   benchmark, not from beating any model on its own turf. Per-task, TRINITY only clearly wins on MMLU.
2. **Math shows no routing benefit at all.** With all three models at ~0.79, TRINITY = best single =
   random. This is a genuine null result, reported as-is, not hidden.
3. **Per-task margins are inside the noise.** TRINITY vs best-single is +0.003 (MMLU) and −0.003 (math) —
   both well inside the ±0.01–0.02 std bars. The honest reading is "ties the specialist per task."
4. **Seed variance in training.** sep-CMA-ES with a noisy binary reward occasionally converged to a bad
   math policy (an earlier seed, math_s0, scored 0.325). MMLU coordinators were robust (0.925–0.95).
5. **The earlier MMLU "failure" (0.55) was a bug, not a finding** — brittle answer-extraction discarded
   correct multi-turn answers. Fixed; MMLU TRINITY moved 0.55 → 0.925. See JOURNAL 2026-06-23.

## 5. What was NOT reproduced / scoped out

- LiveCodeBench (gated loader) and GPQA (gated dataset) were not run on real data, so the paper's 4-task
  suite and the 86.2% LiveCodeBench record are out of scope here.
- Absolute numbers differ from the paper by design (different model pool). We target the **relative**
  invariants (R1/R2/R4), per `SPEC.md` §1.3.
- A separate coding/SWE-bench TRINITY run was scoped and the groundwork located (a sibling project's
  frozen, gold-labelled 3-model candidate pool — where complementarity is real, +12pt oracle over
  best-single) but not executed. See JOURNAL 2026-06-23 (project_harness assessment).

## 6. Reproduce

```bash
source ~/.config/trinity/secrets.env
bash scripts/run_remote.sh train --benchmark math500   # evolve a coordinator on GPU
# rigorous eval: n=120, single baselines averaged over 3 reps
CUDA_VISIBLE_DEVICES=5 python -m trinity.eval --benchmark math500 \
    --theta experiments/math500/full_pilot/best_theta.npy \
    --max-items 120 --single-reps 3 --out experiments/final/math_rigorous.json
python scripts/cost_report.py --ledger cost_ledger.jsonl   # spend
```

## 7. Cost

**$20.89** total (exact, from the token ledger at real Fireworks prices): deepseek $6.56, glm $6.70,
kimi $7.64. Well under the ~$65 projected. Rates: deepseek-v4-pro $1.74/$3.48, glm-5p2 ~$1.40/$4.40,
kimi-k2p6 $0.95/$4.00 per 1M in/out.
