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
