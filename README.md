# TRINITY (open-source replication)

A from-scratch replication of **TRINITY: An Evolved LLM Coordinator** (Xu et al., Sakana AI,
ICLR 2026 — [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)), using **open-source models**
served via Fireworks AI as the coordinated LLM pool.

A tiny coordinator (a **~0.6B** language model whose hidden states encode the query + a
**~10K-parameter** head trained with **separable CMA-ES**) decides, each turn, **which** LLM to
call and **what role** to give it — **Thinker**, **Worker**, or **Verifier** — so the small
coordinator never has to learn the hard skills itself.

> **Start here:** [`AGENTS.md`](AGENTS.md) — goal, environment rules, and the logging protocol.
> Implementation detail lives in [`docs/SPEC.md`](docs/SPEC.md). The lab notebook (mistakes &
> findings) is [`docs/JOURNAL.md`](docs/JOURNAL.md).

## Model pool

| Slot | Fireworks model ID                          |
| ---- | ------------------------------------------- |
| A    | `accounts/fireworks/models/deepseek-v4-pro` |
| B    | `accounts/fireworks/models/glm-5p2`         |
| C    | `accounts/fireworks/models/kimi-k2p6`       |

The 0.6B coordinator encoder and the CMA-ES loop run on a single **NVIDIA H200 (GPU 5)**; the
three LLMs are called over HTTP.

## Setup

```bash
# secrets live OUTSIDE the repo
cp .env.example ~/.config/trinity/secrets.env   # fill in FIREWORKS_API_KEY, then: chmod 600
source ~/.config/trinity/secrets.env

uv venv && source .venv/bin/activate
uv pip install -e .

# confirm the three models answer
python -m trinity.llm.fireworks_client --selftest
```

## Security

No secret ever lives in this repo. SSH key → `~/.ssh/trinity_gpu`; Fireworks key →
`~/.config/trinity/secrets.env`. See [`AGENTS.md`](AGENTS.md) §4.

## Results

**[`docs/RESULTS.md`](docs/RESULTS.md)** — structured scorecard. Headline (rigorous, n=120, baselines
averaged ×3 reps): on the multi-task average TRINITY (**0.858**) beats every fixed single model (best
0.835) and random routing (0.833), so R1/R2 and R4 hold — but **thinly, and cross-task**. The win comes
from picking the right specialist per benchmark: TRINITY clearly beats random on MMLU (0.925 vs 0.875),
while on math all three models cluster at ~0.79 and routing adds nothing over random (an honest null).
Cost tracked via `scripts/cost_report.py` (**$20.89** total).

## Status

Trained + evaluated on math500 + MMLU. Full lab log (every mistake/fix/decision) in
[`docs/JOURNAL.md`](docs/JOURNAL.md).
