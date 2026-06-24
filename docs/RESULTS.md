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
kimi-k2p6 $0.95/$4.00 per 1M in/out. (Plus ~$14 for the oracle-ceiling diagnostic below, tracked
separately in `oracle_cost_ledger.jsonl`.)

## 8. Oracle-ceiling diagnostic (is routing even worth tuning on this pool?)

Recommendation #1 from [`IMPROVEMENTS.md`](IMPROVEMENTS.md), built FP/FN-proof per
[`ORACLE_CEILING_DIAGNOSTIC.md`](ORACLE_CEILING_DIAGNOSTIC.md) (`scripts/oracle_ceiling.py`). For each
held-out query we draw K samples per model, estimate a per-(query,model) solve probability, and compute
the best a **perfect query-conditional router** could reach (`routing_oracle`, winner's-curse-debiased
by split-half cross-fit) vs the best single model. Verdict read off bootstrap CIs, not point estimates.
Raw per-call logs: `experiments/final/oracle_raw_<bench>.jsonl`; reports: `oracle_report_<bench>.json`.

| benchmark | best single | routing oracle | headroom (95% CI) | disagree | verdict |
|---|---|---|---|---|---|
| math500 (K=5) | 0.808 (glm) | **0.856** | **+0.049 [0.005, 0.085]** | 0.29 | **ROUTER_BOUND** |
| mmlu (K=3) | 0.939 (deepseek) | ≥0.939 | +0.025 [0.000, 0.058] (threshold) | 0.50 | INCONCLUSIVE (K<5) |

**Math — ROUTER_BOUND, and it overturns our earlier read.** A perfect router could reach 0.856 vs
best-single 0.808: **+4.9pt of real, achievable headroom** (CI excludes 0). The naive "pick-the-max"
oracle was 0.900; cross-fit stripped 0.044 of winner's-curse inflation. Our trained router captured
**none** of this headroom (it tied best-single and random in §1). So the limit on math is the **router,
not the pool** — the IMPROVEMENTS.md upgrades (warm-start the head, shaped fitness) are warranted. This
corrects the §0/§4 reading of math as "no benefit": there *is* benefit available, our router just misses
it. The diagnostic existed precisely to catch that false-negative.

**MMLU — inconclusive at K=3, near-ceiling in practice.** MMLU was collected at K=3, where the cross-fit
selection half is a single sample and the estimator underflows (it first returned an impossible negative
headroom; the script now floors the oracle at best-single and falls back to the split-free threshold
estimate, flagged `crossfit_reliable=false`). The threshold headroom is small (+2.5pt, CI [0, 5.8pt]) and
straddles 0, so the diagnostic honestly returns INCONCLUSIVE. In practice MMLU is **near-ceiling**:
deepseek dominates (0.94 vs glm 0.79, kimi 0.52) and TRINITY already ≈ best-single (§1, 0.925 vs 0.922),
so the router already captures what little is there. A definitive MMLU cross-fit needs a K≥5 re-collect
(attempted but abandoned after a Fireworks latency spike dropped throughput to ~4 calls/min).

**Net:** on this pool, **math is the place router improvements can pay off** (real headroom our
router misses); MMLU is near-ceiling. Next step from the diagnostic: pursue the warm-start + shaped-
fitness upgrades and validate against the math oracle ceiling (0.856).

## 9. Warm-start + shaped-fitness retrain on math (IMPROVEMENTS.md #2 + #3)

Acting on §8: warm-started the head with a supervised numpy fit of the agent rows against the train-split
oracle matrix (`oracle_matrix_math500_train.json`, K=3, n=200), packed it as the sep-CMA-ES initial mean,
and retrained with shaped training fitness (format bonus + turn penalty + variance reweight; the **eval
stays pure binary**). Config: popsize 8, m_cma 8, gen 12, seed 0. Held-out eval on the same n=120 test
split used for the oracle ceiling. Spend: $27.22 total (this retrain + the train-split label collection).

| | score |
|---|---|
| **TRINITY (warm-start + shaped)** | **0.808** |
| best single (glm-5p2) | 0.817 |
| deepseek-v4-pro | 0.783 |
| kimi-k2p6 | 0.725 |
| random routing | 0.733 |
| *prior router (§1, same test)* | *0.792* |

**Read it as inconclusive, not a win.** The new router scores 0.808 vs the prior router's 0.792 (+1.6pt),
but this is a single new-vs-old point comparison and it sits inside the eval noise. The tell: random
routing scored **0.733 here but 0.792 in the §1 rigorous eval** — same baseline, ~6pt swing — because eval
draws one sample per query (`--single-reps 1`). With the baseline itself moving that much, a +1.6pt router
delta carries no signal. We did **not** run the control (zero-init + pure-binary at the same config) that
would isolate the intervention from CMA-ES seed luck, so **no causal claim**: we cannot say warm-start or
shaping caused the change. The result is still **below best-single (0.817)** and **below the oracle ceiling
(0.856)** — the ~4.9pt of achievable headroom from §8 remains uncaptured. The interventions are
implemented and tested (54 offline tests pass); whether they move the held-out number is unproven.
