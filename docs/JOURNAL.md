# JOURNAL — TRINITY replication lab notebook

This is the running log of **mistakes, findings, and decisions**. See `AGENTS.md` §6 for the
protocol. **Newest entries at the top.** Tag each entry with one or more of:
`#mistake` `#finding` `#decision` `#repro` `#gotcha` `#todo`.

### Entry template

```
## YYYY-MM-DD — short title  #tag #tag
**Context:** what we were doing.
**Expected:** what we thought would happen.
**Actual:** what happened (paste the error/number).
**Root cause:** why.
**Fix / decision:** what we changed and why.
**Follow-up:** anything left open.
```

---

## 2026-06-24 — Warm-start + shaped-fitness retrain on math: inconclusive (within noise)  #finding #decision

**Context:** ran the end-to-end #2+#3 pipeline on the box (`scratchpad/run_warm_shaped.sh`): collect
train-split oracle labels (K=3, n=200) → encode on GPU → numpy fit of the agent head → pack as CMA-ES x0
→ retrain (popsize 8, m_cma 8, gen 12, seed 0) with shaped training fitness → held-out eval on the n=120
test split (same set as the 0.856 oracle ceiling).
**Expected:** if warm-start + shaping helped, the new router beats the prior 0.792 and closes some of the
+4.9pt headroom toward 0.856.
**Actual:** TRINITY 0.808 vs prior 0.792 (+1.6pt). Best single (glm) 0.817, random 0.733. Training best
shaped-fitness 1.0145 at gen 11 (full 12 gens). Spend $27.22 total.
**Root cause / read:** the +1.6pt is inside eval noise. Random routing scored 0.733 here vs 0.792 in the
§1 rigorous eval — same baseline, ~6pt swing — because eval uses `--single-reps 1` (one sample/query).
A moving baseline of that size swamps a 1.6pt router delta.
**Fix / decision:** documented as RESULTS §9 with **no causal claim**. The clean control (zero-init +
pure-binary at the same config) was offered and the user chose to skip it (~$11 to settle borderline
noise), so we do not attribute the change to warm-start or shaping. Result still below best-single (0.817)
and the oracle ceiling (0.856); the achievable headroom from §8 remains uncaptured.
**Follow-up:** if revisited, run the control + raise eval `--single-reps` to ≥3 to shrink the baseline
noise band before claiming any lift. Interventions remain implemented + tested (54 offline tests pass).

---

## 2026-06-23 — Oracle-ceiling diagnostic: math is ROUTER_BOUND (overturns the "math null" read)  #finding #repro #mistake

**Context:** built `scripts/oracle_ceiling.py` (recommendation #1, branch `oracle-ceiling-diagnostic`) to
answer whether routing can help at all on our 3-model pool, FP/FN-proof per the plan. Collected a
per-(query,model) solve matrix (math K=5, MMLU K=3) on the box, $14 (separate `oracle_cost_ledger.jsonl`).

**Result:**
- **math500: ROUTER_BOUND.** Perfect router could reach **0.856** vs best-single 0.808 = **+0.049
  headroom, 95% CI [0.005, 0.085]** (excludes 0). Naive oracle 0.900; cross-fit stripped 0.044 of
  winner's-curse inflation. Models disagree on 29% of queries.
- **mmlu: INCONCLUSIVE at K=3.** Threshold headroom +0.025, CI [0, 0.058]; deepseek dominates (0.94).
  Near-ceiling in practice (TRINITY already ≈ best-single).

**Finding that matters (#finding):** this **overturns** the earlier "math routing gives no benefit"
read. That conclusion came from the trained router tying random/best-single — but the oracle shows there
IS ~+4.9pt of *achievable* headroom on math; our router just captures none of it. So math is limited by
the **router, not the pool** → the warm-start (#2) + shaped-fitness (#3) upgrades are justified, with a
concrete target (math oracle 0.856). The diagnostic earned its keep by catching this false-negative.

**#mistake (caught by the diagnostic's own guard):** MMLU at K=3 first produced an *impossible* negative
headroom (routing_oracle 0.750 < best_single 0.939). Cause: cross-fit splits K in half, so K=3 leaves 1
selection sample/query and the argmax misroutes. Fix: floor the oracle at best_single (a perfect router
can always fall back to the best fixed model), flag `crossfit_reliable=false` when K<5, and base the
verdict on the split-free threshold headroom. Added selftest (e). Cross-fit needs **K>=5** (n_a>=2).

**#gotcha:** the K=6 MMLU re-collect (for a valid cross-fit) was abandoned after Fireworks latency
spiked (throughput fell ~38→4 calls/min, ~5h ETA, no errors). MMLU verdict stands on the threshold
estimate + rigorous-eval evidence; re-collect at K>=5 when the API is healthy for the exact number.

---

## 2026-06-23 — RIGOROUS final eval (n=120, baselines ×3 reps): math null, MMLU win, thin multi-task win  #repro #finding

**Context:** the n=40 per-coordinator numbers were too noisy (same baseline swung 0.45–0.79 across runs).
Ran the definitive eval: n=120 held-out items, each single-model baseline averaged over 3 reps to kill
reasoning-model nondeterminism. Raw: `experiments/final/{math_rigorous,mmlu_rigorous}.json`.

**Result:**

| system | math500 | MMLU | **avg** |
|---|---|---|---|
| **TRINITY** | 0.792 | **0.925** | **0.858** |
| deepseek-v4-pro | 0.747 ± 0.014 | 0.922 ± 0.010 | 0.835 |
| random routing | 0.792 | 0.875 | 0.833 |
| glm-5p2 | **0.794** ± 0.017 | 0.783 ± 0.007 | 0.789 |
| kimi-k2p6 | 0.742 ± 0.018 | 0.539 ± 0.004 | 0.640 |

**Findings (honest):**
- **Multi-task avg: R1/R2 ✅ and R4 ✅, but THIN.** TRINITY 0.858 > best fixed single 0.835 (deepseek)
  > random 0.833. Margins ~0.02.
- **math500: routing gives NO benefit.** TRINITY 0.792 = random 0.792 *exactly*, and ties best single
  (glm 0.794, inside noise). R1/R2 ❌, R4 ❌ for math as a standalone task. Root cause: all three models
  cluster at ~0.74–0.79, so there is no complementarity to exploit — any routing (incl. random) lands at
  ~0.79. This is the SAME thin-complementarity pattern an independent sibling project (project_harness)
  measured on math/proofs (see entry below).
- **MMLU: routing helps.** TRINITY 0.925 > random 0.875 (R4 ✅), edges best single (deepseek 0.922).
  Models are spread (0.54–0.92) → real headroom, captured.
- **The win is CROSS-task, not within-task.** No single model is good at both (deepseek=knowledge,
  glm=math); TRINITY picks the right specialist per benchmark, so its *average* beats any fixed model.
  Within a benchmark of similar models it only matches random.

**Correction to earlier claims (#mistake):** the n=40 math story (TRINITY 0.55 > glm 0.50) that read as a
routing win was small-sample noise; at n=120 it is a tie. RESULTS.md §3 marks the n=40 table superseded.

**Cost (final):** $20.89 exact (deepseek $6.56, glm $6.70, kimi $7.64), well under the ~$65 projection.

**Follow-up:** routing value lives where models genuinely differ. Next levers documented in the entry
below (project_harness) and the classifier-improvement research note.

---

## 2026-06-23 — Assessed sibling repo `project_harness` for reuse (12-agent workflow, every claim verified)  #finding #decision

**Context:** user pointed at `~/Desktop/2026/experiments/project_harness` (a sibling multi-LLM
orchestration project on IMO-ProofBench + SWE-bench Pro) and asked, honestly, whether its data is usable
for TRINITY. Ran a survey→assess→adversarial-verify workflow; all key claims re-checked against files.

**Findings:**
- **Direct data reuse: NO.** Two independent blockers (verified): (1) **model mismatch** — locally-graded
  generator labels cover kimi-k2p6 + glm-5**p1** + gpt-oss-120b; deepseek-v4-pro has zero generator
  grades (it sits on the grading jury), and glm is 5p1 not our 5p2; (2) **domain mismatch** — everything
  is IMO proofs (0–7 LLM-jury) or SWE-bench coding (binary Docker), with **no math500 and no MMLU
  anywhere**.
- **Independent corroboration (valuable):** project_harness measured the SAME thin complementarity on
  math/proofs that we just saw — oracle-union 8/30 vs best-single 7/30 (+1, inside jury MAE), and "no
  live selector beat best-single." Matches our math null result exactly.
- **Real complementarity exists on CODING:** their SWE-bench arena (kimi/deepseek/glm) shows oracle 0.600
  vs best-single 0.480 (+12pt); 6/15 solvable instances solved by exactly one model. Coding is where
  routing would actually pay off.
- **Groundwork is on OUR box:** `/mnt/data/harshal/evo_study/` (same machine) has a frozen 25-instance
  3-model candidate pool (incl. glm-5p2 capture) + gold pass/fail labels + Docker + the SWE-bench eval
  harness. The expensive part (generate + grade candidates) is already done.

**Decision:** a separate coding/SWE-bench TRINITY run was scoped (offline "selection arena": train the
coordinator to route over the frozen, gold-labelled pool for ~$0). **User chose to abandon it** for now
and keep the math/MMLU result clean. Reusable take-aways imported as a prior: routing helps only where
models genuinely differ; report 1-instance margins as noise; value may live in the Verifier role.

**Follow-up:** if we revisit, the offline arena is the cheapest high-value experiment available.

---

## 2026-06-23 — MMLU eval: models SPLIT across tasks → routing headroom confirmed  #repro #finding #decision

**Eval (40 held-out MMLU items; math-trained θ used → TRINITY here is zero-shot transfer):**

| condition | math500 | MMLU |
|---|---|---|
| deepseek-v4-pro | 0.325 | **0.975** ← best on MMLU |
| glm-5p2 | **0.550** ← best on math | 0.725 |
| kimi-k2p6 | 0.275 | 0.600 |
| random routing | 0.350 | 0.675 |
| TRINITY | 0.550 (trained) | 0.850 (zero-shot transfer) |

**Headline finding (#finding):** the pool **splits** — glm-5p2 is the math specialist, deepseek-v4-pro
the knowledge specialist. **No single model wins both.** This is precisely the regime where the paper's
claim lives: a per-task router that picks the right specialist beats any *fixed* single model on the
average. Quick arithmetic with best-per-task routing:
- best fixed single model avg: deepseek (0.325+0.975)/2 = **0.65**; glm (0.55+0.725)/2 = 0.6375.
- per-task-trained TRINITY ≈ per-task best (math→0.55, mmlu→~0.97) → avg ≈ **0.76** > 0.65. R1/R2 holds
  **on the multi-task average** (to be made concrete by training an MMLU coordinator).
- Zero-shot transfer bonus: math-trained θ on MMLU still beat random (0.85 vs 0.675) ✅ R4 again, though
  below deepseek (it was trained to favor glm, the wrong pick for MMLU) — sensible.

**#mistake (minor):** `eval.py --out` crashed writing `experiments/mmlu/...` because the parent dir
didn't exist (results were already printed, so no data lost). Fixed with `parent.mkdir(parents=True)`.

**Decision:** train an MMLU-specific coordinator (paper protocol = one coordinator per task) so we have
a real per-task TRINITY on MMLU, then report the concrete multi-task average vs best single model.

---

## 2026-06-23 — Held-out eval: R4 holds, R1/R2 ties on single task → need multi-task  #repro #finding #decision

**Eval (40 held-out math500 items, trained `full_pilot` θ):**

| condition | acc |
|---|---|
| single deepseek-v4-pro | 0.325 |
| single **glm-5p2** | **0.550** (best single) |
| single kimi-k2p6 | 0.275 |
| **TRINITY (trained)** | **0.550** |
| random routing | 0.350 |

- **R4 ✅** TRINITY (0.55) > random routing (0.35). The learned coordinator is meaningfully better
  than random — it discovered that **glm-5p2 is the math specialist** and routes to it.
- **R1/R2 ❌ (tie, not a win):** TRINITY (0.55) = best single model glm-5p2 (0.55).

**Why this is the EXPECTED outcome, not a failure (#finding):** on a SINGLE benchmark, pure routing
can at best *match* the best model (by always picking it) — there's no headroom to *exceed* it unless
multi-turn Thinker→Worker→Verifier collaboration adds value, which 3 turns / 640 tokens barely
exercises. The paper's "TRINITY > best single model" headline is on the **average across 4 diverse
tasks**, where *no single model wins them all* — that's where per-task routing beats any fixed model.
We replicated the routing mechanism on one task (it found the specialist + beat random); to replicate
the *headline* we need the multi-task setting. Eval is also noisy at n=40 (±~8%), so 0.55 vs 0.55 is a
statistical tie.

**Decision / next:** map single-model strengths across ≥2 benchmarks (math500 + a second where a
*different* model wins, e.g. gpqa/livecodebench). If models split, a per-task router beats best-single
on the average — the real R1/R2 test. Train a coordinator per task (paper protocol) and report the
multi-task average.

---

## 2026-06-23 — Pilot #3 (full_pilot): 12 generations complete, no crashes  #repro #finding

**Result:** First end-to-end training run to **complete all 12 generations** (math500, λ=8, m_cma=8,
3 turns, 64 train tasks) — the timeout-retry + degrade-to-0 fixes held the entire run, zero crashes.

```
gen0 0.094  gen1 0.266  gen2 0.469  gen3 0.500  gen4 0.422  gen5 0.406
gen6 0.297  gen7 0.391  gen8 0.359  gen9 0.469  gen10 0.469  gen11 0.250    (per-gen mean fitness)
```

**Read:** strong early learning (mean 0.094→0.50 over gens 0–3), then the population mean plateaus
and oscillates ~0.25–0.47. `best_fitness` (es.best) = 0.75. `best_theta.npy` saved.

**Caveat (honest):** the cross-generation mean is confounded — common random numbers re-samples a
*different* minibatch each generation, so a harder draw lowers that gen's mean independent of policy
quality (e.g. gen11's 0.250 is likely a hard draw, not regression). So this curve shows "it learned"
but is NOT a clean convergence curve. The decisive measurement is the held-out eval (running now):
TRINITY vs each single model vs random routing on fresh math500 test items (R1/R2/R4).

**Possible next improvements (if eval is inconclusive):** (a) a fixed validation minibatch to make
gen means comparable, (b) larger m_cma to cut reward variance, (c) more generations / σ tuning.

---

## 2026-06-22 — Pilot #2 (CRN): J climbs, then crashes on httpx.ReadTimeout  #repro #mistake #finding

**Result (the headline so far):** with common random numbers, **sep-CMA-ES learns** — clean upward J:

```
gen0 mean=0.141 best=0.375    gen1 mean=0.203 best=0.375    gen2 mean=0.438 best=0.625
```

Mean fitness tripled and best improved 0.375→0.625 in 3 generations on math500 — the core
replication claim (the trained coordinator improves over evolution) is demonstrated in principle.
`best_theta.npy` + `history.json` saved each generation, so gen-2's θ (fitness 0.625) survived.

**#mistake — run crashed at gen 3 on `httpx.ReadTimeout`.** My Fireworks client retried only on HTTP
status codes (429/5xx), not on network timeouts / transport errors, so one slow reasoning call (>120s)
raised an uncaught `ReadTimeout` that propagated through `asyncio.gather` and killed the whole run.
(The `python | tee` pipeline masked it as exit 0 — `tee` succeeded even though python died.)

**Fixes (#decision):**
1. `FireworksPool.chat` now catches `httpx.TimeoutException`/`TransportError` and retries them
   (transient blips are normal over thousands of calls). Timeout 120→180s, retries 4→6.
2. `evaluate_candidate` uses `gather(return_exceptions=True)` — a trajectory that exhausts retries
   degrades to **reward 0** and logs a warning, instead of crashing the generation/run.
3. Next run uses `nohup` (survives an ssh drop). **Detach gotcha (root-caused):** `nohup cmd > log &`
   inside a long `a && b && read KEY && nohup ... &` chain silently failed because `&` binds the
   *whole* `&&` list — so the chain (including `read KEY`) ran in a backgrounded subshell whose stdin
   is `/dev/null`, the `read` hit EOF, the chain aborted before launching python, and `$!` was just
   the dead subshell. Fix: keep the key-`read` in the foreground and background only the launch with a
   brace group: `... && export KEY && { nohup python ... > log 2>&1 </dev/null & }`. Verified the log
   is created and training proceeds.

**Lesson:** a long API-bound training loop must treat transient network errors as expected, not fatal.
CPU smoke ladder still 4/4 green after the fixes.

---

## 2026-06-22 — Pilot #1: flat J → root cause = per-candidate minibatch noise → CRN fix  #mistake #finding #decision

**Context:** First real pilot (math500, λ=6, m_cma=4, 12 gens). Ran clean across 5 generations on
GPU 5 (no crashes, per-candidate fitness varied 0.0–1.0), but **mean fitness did not climb**:

```
gen0 mean=0.375  gen1 0.542  gen2 0.375  gen3 0.375  gen4 0.250   (max stuck at 1.0 = lucky minibatch)
```

**Root cause (#mistake in my training loop):** `minibatch_fn(i)` drew a *fresh random* minibatch for
*each candidate* within a generation. With `m_cma=4` the per-candidate fitness is a mean of 4
Bernoulli draws (std ≈ 0.25) AND each candidate saw *different tasks* — so sep-CMA-ES was ranking
candidates largely by task-luck, not policy quality. The "best=1.0" was one candidate that drew 4
easy problems, not a good coordinator. This is precisely the binary-reward variance risk flagged in
SPEC §0.4 and the review (#5).

**Fix / decision (#decision):** **Common Random Numbers (CRN)** — score *all* candidates in a
generation on the *same* minibatch (re-sampled across generations). Standard ES variance-reduction:
fitness *differences* now reflect policy differences, not which tasks were drawn. Also raised
`m_cma 4 → 8` to halve the reward-estimate std. Re-launched as `pilot_crn`. This is a deliberate
[OUR CHOICE] deviation from the paper's "re-sample per replication" phrasing; documented because it
materially changes the optimizer's signal.

**Lesson:** for noisy-reward ES, *how* you sample the fitness minibatch matters as much as the
optimizer. Watch whether `pilot_crn` shows a cleaner upward J trend.

---

## 2026-06-22 — GPU env up; full smoke ladder S1-S8 PASS; pilot training launched  #repro #finding #gotcha

**Context:** Provisioned the H200 box and ran the GPU/network smoke rungs, then launched training.

**Env:** `uv` venv on the box, **torch 2.12.1+cu130**, transformers 5.12.1, numpy 2.5.0, cma. With
`CUDA_VISIBLE_DEVICES=5`, `torch.cuda.device_count()==1` (correctly pinned to our H200).

**Smoke ladder (GPU/net rungs) — ALL PASS on GPU 5:**
- **S1**: Qwen3-0.6B loads, `hidden_size=1024`, `layers=28`, encode deterministic, `‖h‖=1.0000`.
- **S2**: SVF `num_scales=7168` — **exactly the predicted 7×1024, confirmed on the real
  checkpoint** (resolves the paper's 9,216 discrepancy for our model). Identity round-trips
  (`max|Δ|=1.2e-3` bf16), perturb changes weights, `reset()` exact.
- **S6**: all 3 Fireworks models answer live with `reasoning_effort=minimal`.
- **S8**: end-to-end fitness produced within the call budget.

**#gotcha — detached launch quoting.** First `nohup`/`setsid` launches failed silently (log file
never created) due to nested single/double-quote + `\$HOME` escaping across the ssh boundary, plus
SIGHUP timing. **Fix:** run training via a local-background `ssh ... | tee` (key fed on stdin →
remote env, never in argv/disk); foreground sanity run first confirmed the loop works
(`gen0 best=0.500`, n=13312, 85s for pop3×m2).

**#finding — uniform-policy argmax degenerates to THINKER.** With `W=0`, argmax over uniform role
logits always picks role index 0 = THINKER (ROLE_ORDER[0]), so a never-trained coordinator under
argmax produces no Worker answer → reward 0. Training uses `sample=True` so it explores; this is
expected, not a bug. Eval uses argmax (post-training, when W is non-trivial).

**Pilot config (running):** math500, λ=8, m_cma=6, T=8, max_items=64, max_turns=3, max_tokens=1024
→ budget ≈ 384 atomic evals. Watching whether `J` rises before committing to a full-budget run.

---

## 2026-06-22 — Core implemented; CPU smoke rungs green; review bugs fixed  #repro #mistake #finding

**Context:** Implemented all modules (coordinator, roles, orchestration, sep-CMA-ES, train/eval)
via a parallel build + adversarial integration review, then validated the CPU smoke ladder.

**Integration review caught 3 real bugs (fixed before any GPU/LLM spend):**
- **#mistake P0 — LiveCodeBench reward was identically 0.** `reward._run_one_test` read stdin from
  `test["stdin"]` but `dataset.py` emits `{"input":..., "output":...}`, so every code test ran on
  empty stdin → reward 0 for every candidate → CMA would optimize against a dead signal. Fixed to
  read `test.get("stdin", test.get("input",""))` and trigger on `input`/`output` keys.
- **#mistake P1 — `trinity.optim` couldn't import without pycma.** `sep_cmaes.py` re-raised the
  `cma` ImportError at module top. Deferred it into `_import_cma()` (cma is only needed to build
  the optimizer at train time).
- **#mistake P2 — choice-letter extractor matched prose.** `"A nice approach"` → `"A"`. Tightened
  the regexes + restricted the fallback to a final standalone-letter line. Also fixed plain-text
  fraction extraction (`"1/2"` was read as `"2"`).

**Smoke ladder (SPEC §11), CPU rungs run locally — ALL PASS:**
- S3 params pack/unpack round-trip, `n_total=13312`, head `(6,1024)`, `n_svf=7168`.
- S4 multi-turn termination + worker-guarded Verifier-ACCEPT + fail-safe REVISE.
- S5 reward checkers (math incl. fractions, MMLU/GPQA letters, code pass@1 with stdin).
- S7 sep-CMA-ES maximizes a synthetic objective; `popsize(13312)=33` confirmed.

**Follow-up:** provision GPU box env, run GPU rungs S1 (encoder/penultimate), S2 (SVF identity +
real scale count), S6 (live pool), S8 (end-to-end fitness), then launch sep-CMA-ES training on GPU 5.

---

## 2026-06-22 — Fireworks reasoning-effort mapping resolved  #finding #decision

**Context:** SPEC left "minimal reasoning effort" → API param unspecified (open item #12/#16).
**Finding:** all 3 models accept `reasoning_effort` ∈ {none, low, medium, high} (HTTP 200).
**Decision:** map "minimal" → `reasoning_effort: "low"` in `FireworksPool.chat`; configurable.

---

## 2026-06-22 — Paper → SPEC, with verified facts & review corrections  #finding #decision #repro

**Context:** Ran a 9-agent deep read of the paper → `docs/SPEC.md` (+ `PAPER_NOTES.md`,
`SPEC_REVIEW.md`). Then grounded the risky numbers against ground truth.

**Findings / corrections (SPEC §0 is authoritative):**
- **Qwen3-0.6B real config:** `hidden_size=1024` (d_h CONFIRMED), **28 layers** (2nd-to-last =
  index 26), GQA (16 q / 8 kv heads, head_dim 128 → `q_proj` is 1024×2048, `o_proj` 2048×1024),
  SwiGLU `intermediate_size=3072`, tied embeddings, bf16. Every linear matrix has min-dim 1024 →
  **1024 SVs each**.
- **SVF count mismatch (#mistake-averted):** paper states 9,216 SVF scales (=9×1024), but a Qwen3
  layer has only **7** linear matrices → 7×1024 = **7,168**. The paper's 9,216 does not map onto
  Qwen3's matrix set. **Decision:** SVF all 7 matrices of layer 26 → 7,168 scales (init 1.0),
  documented delta. Smoke test **S2 must print the real count** and assert θ matches.
- **CMA λ arithmetic error caught by review:** spec body said λ=34; correct is
  `⌈4+3·ln(13312)⌉ = ⌈32.49⌉ = 33`. Budget `B_env = 16·33·60 = 31,680` (body's "34,560" was wrong).
- **Totals (ours):** head `6×1024 = 6,144` + SVF `7,168` = **n = 13,312** trainable (CMA dim).
- **Decisions:** L2-normalize `h` before the head (σ₀ stability); Verifier can't ACCEPT before a
  Worker output exists; MT-Bench is report-only (never binarized into reward); single-model
  baselines run at 20,480 tokens (5×) for fair R1/R2; disk-cache LLM calls.
- **Open risk:** block-ε-separability was shown on the paper's 7-agent pool; it may NOT transfer
  to our 3-model pool, so **R8 (CMA > SFT > RS > REINFORCE) is a hypothesis to test**, not assumed.

**Follow-up:** implement M0–M4, pass the S1–S8 smoke ladder, then run head + sep-CMA-ES on GPU 5.

---

## 2026-06-22 — Remote H200 box inventory  #finding

**Context:** Inventoried `trinity-gpu` (read-only) before writing the remote setup path.

**Findings:**
- **Ubuntu 24.04**, **192 vCPU**, **~2 TB RAM**. `$HOME` = `/mnt/data/harshal` on a 12 TB array
  with **3.2 TB free** — ample for HF model caches.
- **8× NVIDIA H200 NVL (143 GB each)**, driver `595.71.05`. **We use index 5 only.**
- Python **3.12.3** present; **no `uv`, no `conda`, no `torch`** system-wide → `setup_remote.sh`
  installs `uv` and builds a project `.venv`.
- Network: `huggingface.co` → 200 (model downloads OK). `api.fireworks.ai` root → 404, which is
  expected (the API lives under `/inference/v1`; root has no handler). Not a problem.
- No pre-existing `~/trinity`; we sync there fresh.

**Decision:** default `TRINITY_REMOTE_DIR=$HOME/trinity` (= `/mnt/data/harshal/trinity`),
`HF_HOME` under the project dir to avoid polluting shared `$HOME`.

---

## 2026-06-22 — Project bootstrap & environment verification  #finding #decision

**Context:** Kicking off the replication. Verified the full toolchain before writing code.

**Findings:**
- **GPU box reachable.** `ssh trinity-gpu` works. `nvidia-smi -i 5` reports **NVIDIA H200 NVL,
  143771 MiB, idle**. We are allocated **GPU 5 only** — all CUDA work pins
  `CUDA_VISIBLE_DEVICES=5`.
- **All three Fireworks models answer** a chat completion (HTTP 200):
  `accounts/fireworks/models/deepseek-v4-pro`, `.../glm-5p2`, `.../kimi-k2p6`.
- **GitHub:** authed as `harrrshall` with `repo` scope; repo will be created **private**.

**Decisions:**
- Secrets live **outside the repo**: SSH key at `~/.ssh/trinity_gpu` (600) behind the
  `trinity-gpu` host alias; Fireworks key at `~/.config/trinity/secrets.env` (600), read via
  `FIREWORKS_API_KEY`. `.gitignore` blocks all secret patterns + the 11 MB paper PDF.
- Replication target is the paper's **relative** claims (TRINITY > best single model > random
  routing; sep-CMA-ES > RL/IL/random), since our open-source model pool differs from the
  paper's. See `AGENTS.md` §1.

## 2026-06-22 — Fireworks account model-list endpoint 500s  #gotcha

**Context:** Tried `GET /inference/v1/models` to enumerate available models.
**Actual:** `HTTP 500 — "Error listing deployed models"`.
**Root cause:** That account-level listing endpoint is flaky / not enabled for this key; it is
not needed.
**Fix / decision:** Probe model IDs directly with a tiny `chat/completions` call instead. All
three target IDs returned 200, so we hardcode them in `configs/models.yaml`. Do **not** depend
on the list endpoint.

---

## 2026-06-23 — Parallel multi-seed training across free GPUs  #decision #repro

**Context:** User cleared use of all free GPUs (1,4,6,7); GPU 5 keeps its run. Launched stronger
coordinators with seed replicates for an error-barred multi-task result.

| GPU | run | task | seed | config |
|---|---|---|---|---|
| 1 | math_s0 | math500 | 0 | λ8, m_cma10, T14, 4 turns, 768 tok, 80 items |
| 4 | math_s1 | math500 | 1 | same |
| 5 | mmlu_pilot | mmlu | — | basic (earlier run, still finishing) |
| 6 | mmlu_s0 | mmlu | 0 | strong |
| 7 | mmlu_s1 | mmlu | 1 | strong |

**Caveat (#finding):** workload is **API-bound, not GPU-bound** (GPUs idle ~0% util waiting on
Fireworks). 5 parallel processes share the Fireworks rate limit → ~40 concurrent calls; 429s are
retried (robustness fix), so throughput is shared, not 5×. Real benchmarks limited to math500 + mmlu
(gpqa gated, livecodebench loader → toy fallback; both need extra work to enable real data).

**Plan:** when all finish, eval each coordinator on its held-out test set, then report
TRINITY-per-task (mean±std over seeds) vs best single fixed model on the math+mmlu average = the R1/R2
headline test.

---

## 2026-06-23 — Cost tracking added  #decision #finding

**Context:** User asked to track Fireworks cost. No usable billing API (probed: `/v1/usage`→404,
`/v1/accounts/usage`→403), but every chat response carries exact token counts, so we price from tokens.

**Built:**
- `scripts/cost_report.py` — `--ledger` (exact, from recorded tokens) or `--estimate` (from run configs).
- `FireworksPool` now appends `{model, prompt_tok, completion_tok}` to `$TRINITY_COST_LEDGER` per call
  (best-effort JSONL) → future runs/evals are tracked exactly. The 5 in-flight runs predate this, so
  they're estimated.

**Empirical fact:** reasoning models fill ~all of `max_tokens` on completion (glm used 400/400),
so completion_tokens ≈ max_tokens; prompt grows with the multi-turn transcript (~650 avg).

**Estimate (ASSUMED prices, blended ~$0.67/1M in, $2.10/1M out — NOT confirmed):**
- Spent so far (pilots + 2 evals): **≈ $5**.
- Projected when the 5 current runs finish: **≈ $34 total** (~24M tokens, ~17k calls).
- The 4 strong parallel runs dominate (~$6.4 each).

**TODO:** get real per-model Fireworks rates from the dashboard to convert estimate → exact. Report a
live cost line at each monitoring checkpoint (scale per-run cost by generations completed).

---

## 2026-06-23 — Real Fireworks prices → cost ~doubled  #finding

Web search gave real serverless rates ($/1M, in/out): deepseek-v4-pro **1.74 / 3.48**,
kimi-k2p6 **0.95 / 4.00**, glm-5p2 **1.40 / 4.40** (GLM-5.1 proxy; 5.2 not separately listed).
Output is ~$4/M, so the projection rose from ~$34 (assumed) to **~$65 total** (~$10 spent so far;
the 4 strong parallel runs ~$12 each). All 5 runs confirmed RUNNING. Caching (~50% off cached input)
ignored, so this slightly over-estimates.

---

## 2026-06-23 — MMLU underperformance root-caused: brittle extraction + bad routing  #mistake #finding #decision

**Diagnostic** (dumped argmax trajectories of the mmlu_pilot coordinator) revealed two compounding bugs:
1. **Extraction loses correct answers.** A worker derives "degree 2" (correct, =B) but emits no clean
   letter; `_final_answer` returns that verbose worker output → `extract_choice_letter`→None → reward 0,
   even though the system found the answer. This penalizes multi-turn answers; the single-model baseline
   (one clean direct call) extracts fine → unfair 0.55-vs-0.95 gap.
2. **Coordinator routes to kimi/glm, NOT deepseek** (the MMLU champion). It never learned deepseek is best
   for MMLU — because bug #1 made the *training* reward noisy (correct answers scored 0), corrupting the
   signal the optimizer needed to distinguish deepseek > kimi.

**Fix #1 (done):** `score()` now scores the MOST RECENT turn with an *extractable* answer (falling back to
final_answer), applied equally to all conditions. Re-evaluating to quantify how much of the gap was
extraction vs routing.

**Implication:** the extraction bug hurt BOTH eval scoring AND training quality. Cleanly fixing it likely
needs a re-train (so the optimizer sees an honest reward and learns to route to the right specialist).
Caveat for choice tasks: models often answer with the VALUE ("degree 2") not the LETTER ("B"); fully
fixing that needs answer-format prompting (task-aware), noted as a follow-up.

---

## 2026-06-23 — Extraction fix → MMLU TRINITY 0.55→0.95 (ties best single); multi-task R1/R2 holds  #repro #finding

**mmlu_s1 re-eval with fixed extraction (40 held-out items):**
deepseek 0.975 | glm 0.750 | kimi 0.600 | random 0.850 | **TRINITY 0.950**.
TRINITY jumped 0.55→**0.95**, now a statistical tie with the best single model (deepseek 0.975) — the
0.40 gap was almost entirely the extraction bug, confirming the diagnosis. R4 holds (0.95>0.85).

**Multi-task headline (the paper's core claim) — reproduced on our open-source pool:**
- best FIXED single model avg ≈ **0.65** (deepseek 0.33/0.975; glm 0.55/0.75).
- per-task TRINITY avg ≈ **0.75** (math 0.55, mmlu 0.95) → **TRINITY > any single fixed model on the
  multi-task average.** No single model wins both tasks; routing to the per-task specialist does.
- Per-task, TRINITY ties the best specialist (math≈glm, mmlu≈deepseek) and beats random on both — the
  expected single-task ceiling for routing.

**Status:** strong math coordinators at 13/14; re-evaluating math with the fixed extraction to finalize
the table. Exact ledger cost so far ~$1.3 (evals); total spend ~$13.

---

## 2026-06-23 — Structured results + rigorous eval launched  #decision #repro

Per user request (document everything + structured output):
- `scripts/results_table.py` aggregates all `experiments/**/eval*.json` → structured Markdown table +
  `experiments/results.json` (machine-readable). `docs/RESULTS.md` is the human report (linked from README).
- Current aggregated verdict (40-item evals): **R1/R2 ✅ 0.750>0.639, R4 ✅ 0.750>0.558**, with caveats:
  math seed variance (math_s0 failed at 0.325), n=40 eval noise (single baselines swing 0.45–0.70).
- **Rigorous eval running** (GPU5): n=120, single baselines ×3 reps (kills reasoning-model
  nondeterminism), best math (full_pilot) + best MMLU (mmlu_s1) coordinators → definitive numbers.
- Cost ~$22 (ledger-tracked). No GPU was empty (other tenants), but evals are light (~4 GB) so they
  coexist on a shared H200.
