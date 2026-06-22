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
