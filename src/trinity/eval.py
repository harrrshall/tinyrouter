"""Entrypoint: evaluate a trained coordinator + baselines on a benchmark.

Reports the relative invariants from SPEC §1.3:
  - TRINITY (trained coordinator, argmax) vs
  - each single model alone (one direct Worker turn) [R1, R2] vs
  - random routing (random agent+role each turn) [R4].

Usage:
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m trinity.eval --benchmark math500 \
        --theta experiments/math500/run/best_theta.npy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .llm.fireworks_client import FireworksPool
from .orchestration import reward as R
from .orchestration.dataset import load_tasks
from .orchestration.session import run_trajectory
from .types import ROLE_ORDER, Role

_REPO = Path(__file__).resolve().parents[2]


class RandomPolicy:
    """Random (agent, role) each turn — the R4 routing baseline (no GPU)."""

    def __init__(self, n_models: int, seed: int = 0):
        self.n_models = n_models
        self.rng = random.Random(seed)

    def decide(self, transcript_text, *, sample=False, rng=None):
        return self.rng.randrange(self.n_models), self.rng.choice(ROLE_ORDER)


async def _score_policy(tasks, policy, pool, pool_models, *, sample, **run_kwargs) -> float:
    import httpx

    async with httpx.AsyncClient() as cli:
        trajs = await asyncio.gather(
            *[
                run_trajectory(t, policy, pool, pool_models, sample=sample, client=cli, **run_kwargs)
                for t in tasks
            ]
        )
    return float(mean(R.score(t) for t in trajs))


async def _score_single_model(tasks, pool, model, benchmark, *, max_tokens, reasoning) -> float:
    """Baseline: ask one model directly (one Worker-style turn), score its answer."""
    import httpx

    from .roles.prompts import build_messages

    async with httpx.AsyncClient() as cli:
        async def one(task):
            msgs = build_messages(Role.WORKER, task.prompt, [])
            res = await pool.chat(model, msgs, max_tokens=max_tokens, temperature=0.0,
                                  reasoning=reasoning, client=cli)
            return R.score_text(benchmark, res.text, task.answer)

        scores = await asyncio.gather(*[one(t) for t in tasks])
    return float(mean(scores))


async def evaluate(args) -> dict:
    pool = FireworksPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    tasks = load_tasks(args.benchmark, "test", max_items=args.max_items, seed=args.seed)
    print(f"[eval] benchmark={args.benchmark}  {len(tasks)} test tasks  pool={pool_models}")
    run_kwargs = dict(max_turns=args.max_turns, max_tokens=args.max_tokens, reasoning=args.reasoning)

    results: dict[str, float] = {}

    # --- single-model baselines (R1/R2) ---
    for m in pool_models:
        reps = [await _score_single_model(tasks, pool, m, args.benchmark,
                                          max_tokens=args.max_tokens, reasoning=args.reasoning)
                for _ in range(max(1, args.single_reps))]
        s = float(mean(reps))
        results[f"single::{m}"] = s
        if len(reps) > 1:
            sd = (sum((x - s) ** 2 for x in reps) / len(reps)) ** 0.5
            results[f"single_std::{m}"] = sd
            print(f"  single  {m:20s} = {s:.4f} ± {sd:.4f}  (reps={reps})")
        else:
            print(f"  single  {m:20s} = {s:.4f}")

    # --- TRINITY trained coordinator (argmax) ---
    cfg = yaml.safe_load(Path(args.config).read_text())["coordinator"]
    print("[eval] building coordinator on GPU...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cfg["encoder_model"], device=cfg.get("device", "cuda:0"),
        dtype=cfg.get("dtype", "bfloat16"), target_layer=cfg["svf"]["target_layer"],
        svf_matrices=cfg["svf"].get("matrices"), n_models=n_models,
        l2_normalize=cfg["hidden_state"].get("l2_normalize", True),
    )
    theta = np.load(args.theta)
    policy.configure(theta, spec)
    s_trinity = await _score_policy(tasks, policy, pool, pool_models, sample=False, **run_kwargs)
    results["TRINITY"] = s_trinity
    print(f"  TRINITY (trained)        = {s_trinity:.4f}")

    # --- random routing (R4) ---
    rand = RandomPolicy(n_models, seed=args.seed)
    s_rand = await _score_policy(tasks, rand, pool, pool_models, sample=False, **run_kwargs)
    results["random_routing"] = s_rand
    print(f"  random routing           = {s_rand:.4f}")

    best_single = max(results[k] for k in results if k.startswith("single::"))
    invariants = {
        "R1/R2 TRINITY > best single model": s_trinity > best_single,
        "R4 TRINITY > random routing": s_trinity > s_rand,
        "best_single": best_single,
    }
    out = {"benchmark": args.benchmark, "results": results, "invariants": invariants}
    print("[eval] invariants:", json.dumps(invariants, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TRINITY + baselines")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--theta", required=True, help="path to trained best_theta.npy")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=100, dest="max_items")
    ap.add_argument("--single-reps", type=int, default=1, dest="single_reps",
                    help="average each single-model baseline over K runs (cuts nondeterminism noise)")
    ap.add_argument("--max-turns", type=int, default=5, dest="max_turns")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
