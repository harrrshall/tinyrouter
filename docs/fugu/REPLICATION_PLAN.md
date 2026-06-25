# OpenFugu replication on the open-source pool: build plan + status

Branch: `openfugu-replication`. Goal: replicate OpenFugu (trotsky1997/OpenFugu),
that is, add the **Conductor / Fugu-Ultra** orchestration tier that TinyRouter
lacks, over the existing open-source Fireworks pool (`deepseek-v4-pro`,
`glm-5p2`, `kimi-k2p6`), with cost tracked and false positives/negatives guarded.

Research basis: [`FUGU_REPLICATION_RESEARCH.md`](./FUGU_REPLICATION_RESEARCH.md),
sources in [`REFERENCE_INDEX.md`](./REFERENCE_INDEX.md).

## Status

| Piece | State |
| --- | --- |
| Workflow schema + strict parse-gate + executor (access-list topology, bounded recursive self-call) | **built, offline-tested** (`src/trinity/fugu/workflow.py`) |
| Two-stage Conductor reward on the FIXED grader; pure-binary eval | **built, offline-tested** (`src/trinity/fugu/reward.py`) |
| Conductor policy: prompted baseline + stub; trained-LM seam | **built** (`src/trinity/fugu/conductor.py`) |
| GRPO math (group-normalized, no KL) + rollout/loop + cost cap | **built, offline-tested** (`src/trinity/fugu/grpo.py`) |
| Honest eval harness (pure binary, multi-rep, oracle-ceiling-compatible) | **built** (`src/trinity/fugu/eval.py`) |
| API-cost accounting: per-run pricing, running meter, pre-run estimators | **built, offline-tested** (`src/trinity/fugu/cost.py`) |
| Offline test suite (21 tests, no network/GPU/spend) | **passing** (`tests/test_fugu_*.py`) |
| Trained-LM backend (HF/TRL GRPO on the H200) | **pending** (the `PolicyBackend` seam; needs the remote box + budget) |
| The actual paid GRPO training run + rigorous eval | **pending, gated on a budget decision** |

Nothing built so far spends any money or needs a GPU. Everything is exercised by
stubbed pools and conductors.

## Architecture

Two tiers behind one endpoint, mirroring Fugu's own split (and OpenFugu):

* **Fugu (base) = the existing TinyRouter** CMA-ES router (pick one model + role).
  Already in this repo; the cheap fast path.
* **Fugu-Ultra = the Conductor** (this branch): a policy that emits a
  natural-language workflow over the pool and synthesizes the answer. Trained by
  GRPO. The heavy path.

## Module map

```
src/trinity/fugu/
  workflow.py   Workflow / WorkflowStep / WorkflowRun; parse_workflow (the
                format gate); run_workflow + propose_and_run (executor with
                access-list context + bounded recursive self-call). Per-model
                token accounting on every run (for exact cost).
  reward.py     training_reward (parse-gate -> 1.0/0.5) for GRPO;
                is_correct (PURE 0/1) for reporting; committed_answer (the
                multi-step FP/FN-recovery pick). All correctness via the shared
                trinity.orchestration.reward.score_text (the FIXED grader).
  conductor.py  Conductor protocol; PromptedConductor (zero-training baseline
                over a Fireworks model); StubConductor (offline tests). The
                trained HF policy implements the same protocol.
  grpo.py       group_advantages (no KL); collect_group (rollouts + reward +
                cost meter); train (loop skeleton calling a PolicyBackend.update);
                CostCapExceeded guard.
  cost.py       PRICES; run_cost; CostMeter (running spend + cap);
                estimate_grpo_cost / estimate_eval_cost (pre-run projections).
  eval.py       evaluate (pure binary, multi-rep, emits per-query 0/1 for the
                oracle-ceiling diagnostic) + cost.
```

Reused unchanged: `llm/fireworks_client` (pool + cost ledger), `roles/postprocess`,
`orchestration/reward` (the grader), `types`, and `scripts/oracle_ceiling.py`
(the FP/FN-proof ceiling + bootstrap CIs).

## How false positives / negatives are avoided

1. **One grader.** Correctness is decided ONLY by `orchestration.reward.score_text`,
   the same de-bugged extractor used by `eval.py` and the oracle diagnostic. No
   new grading path is introduced, so the prose-"A" false positive and the
   LiveCodeBench-reward-0 false negative the JOURNAL records cannot reappear here.
2. **Training vs eval are separate.** `training_reward` (parse-gate + partial
   credit) drives GRPO; `is_correct` (pure 0/1, no partial, no leniency) is the
   only number eval reports. A shaped reward can never inflate a reported score.
3. **Parse-gate is a hard floor.** A malformed workflow scores 0; the proposal
   text is never executed (lists parsed with `ast.literal_eval` only).
4. **Answer-format hints + committed-answer recovery** keep a correct answer from
   being lost to an unparseable shape (a false negative).
5. **Multi-rep eval + oracle ceiling.** `evaluate(reps>1)` denoises the single-
   sample swing (RESULTS.md saw random routing move about 6 points on one
   sample); the per-query 0/1 it emits feeds `scripts/oracle_ceiling.py` for the
   winner's-curse-debiased ceiling and bootstrap-CI verdict.

## Cost accounting

Every worker and conductor call already lands in the shared cost ledger (set
`TRINITY_COST_LEDGER`; aggregate with `scripts/cost_report.py`). On top of that:

* Each `WorkflowRun` carries exact per-model `(prompt, completion)` token totals
  (including recursion and the conductor's own generation), priced by
  `cost.run_cost`.
* `cost.CostMeter` tracks running spend and **aborts a loop at a `--max-cost-usd`
  cap** (`grpo.train` and `eval.evaluate` both honor it), so a run cannot
  silently overspend.
* `cost.estimate_grpo_cost` / `estimate_eval_cost` project spend BEFORE a run.

Projected Fireworks API spend (conductor served locally on the H200, so only
worker calls cost API money; assumes ~2.5 steps/workflow, ~1.2k prompt + ~0.8k
completion tokens/worker call):

| Run | Rollouts | Worker calls | Projected API $ |
| --- | --- | --- | --- |
| GRPO Phase 0 smoke (G8 x 5it x 4q) | 160 | 320 | ~$1.5 |
| GRPO Phase 1 small (G16 x 40it x 4q) | 2,560 | 6,400 | ~$31 |
| GRPO paper-scale (G64 x 200it x 4q) | 51,200 | 128,000 | ~$615 |
| Eval 120 x 1 rep | 120 | 300 | ~$1.4 |
| Eval 120 x 3 reps | 360 | 900 | ~$4.3 |

Paper-scale GRPO is expensive because the bottleneck is paid worker rollouts, not
GPU. Options to cut it: smaller G, fewer iterations, a cheaper worker subset
during training, or caching worker answers for repeated (query, subtask) pairs.

## Running the real training (remote H200, paid)

The only missing piece is a trainable `PolicyBackend` (an HF model fine-tuned by
GRPO). Recipe:

1. On `trinity-gpu` (GPU 5 only), load a base: Qwen3-0.6B for the cheap Phase 0
   loop validation, then Qwen3.5-4B for the real Conductor.
2. Implement `PolicyBackend` (see `grpo.py`): `propose` generates a workflow and
   returns token log-probs; `update` applies the GRPO policy-gradient step
   (advantages from `group_advantages`, no KL). TRL or verl supplies the update.
3. `source ~/.config/trinity/secrets.env`, set `TRINITY_COST_LEDGER`, and run
   `grpo.train(..., cfg=GRPOConfig(max_cost_usd=<cap>))` so spend is capped.
4. Evaluate with `fugu.eval.evaluate(reps=3)`, then feed `per_query_binary` to
   `scripts/oracle_ceiling.py` for the honest ceiling and CI verdict.

Phase order: Phase 0 (cheap, prove the loop) before Phase 1 (real Conductor)
before paper-scale. Stop at the first phase that fails to clear random routing.

## Open decision

The paid GRPO training run is the one step that costs money and needs the remote
GPU. Pick a budget tier (Phase 0 ~$1.5, Phase 1 ~$31, paper-scale ~$615) before
launching. Until then the implementation stands complete and offline-verified.
