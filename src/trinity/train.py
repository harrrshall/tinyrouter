"""Entrypoint: evolve the coordinator (linear head + SVF scales) with sep-CMA-ES.

One coordinator is trained per benchmark (SPEC §6.1). Each candidate θ is scored
by the mean binary reward over a freshly-sampled minibatch of `m_cma` train tasks.

Usage (on GPU 5 via scripts/run_remote.sh, or directly):
    source ~/.config/trinity/secrets.env
    CUDA_VISIBLE_DEVICES=5 python -m trinity.train --benchmark math500 \
        --config configs/trinity.yaml --models configs/models.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from .coordinator import params as P
from .coordinator.policy import CoordinatorPolicy
from .llm.fireworks_client import FireworksPool
from .optim.fitness import evaluate_population
from .optim.sep_cmaes import SepCMAES, default_popsize
from .orchestration.dataset import load_tasks, sample_minibatch

_REPO = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


async def train(args) -> dict:
    cfg = _load_yaml(args.config)
    cc = cfg["coordinator"]
    sc = cfg["sep_cmaes"]
    sess = cfg.get("session", {})

    pool = FireworksPool(args.models)
    pool_models = list(pool.models)
    n_models = len(pool_models)

    print(f"[train] benchmark={args.benchmark}  pool={pool_models}")
    print("[train] building coordinator on GPU (this loads Qwen3-0.6B)...")
    policy, spec = CoordinatorPolicy.build(
        model_name=cc["encoder_model"],
        device=cc.get("device", "cuda:0"),
        dtype=cc.get("dtype", "bfloat16"),
        target_layer=cc["svf"]["target_layer"],
        svf_matrices=cc["svf"].get("matrices"),
        n_models=n_models,
        n_roles=cc["head"].get("n_roles", 3),
        l2_normalize=cc["hidden_state"].get("l2_normalize", True),
    )
    assert spec.n_svf == int(policy.svf.num_scales), (
        f"spec.n_svf={spec.n_svf} != svf.num_scales={policy.svf.num_scales}"
    )
    print(f"[train] θ dimension n = {spec.n_total} (head {spec.n_head} + SVF {spec.n_svf})")

    tasks = load_tasks(args.benchmark, "train", max_items=args.max_items, seed=args.seed)
    print(f"[train] loaded {len(tasks)} train tasks")

    popsize = args.popsize or sc.get("population_size") or default_popsize(spec.n_total)
    m_cma = args.m_cma or sc.get("m_cma", 16)
    generations = args.generations or sc.get("generations", 60)
    sigma0 = sc.get("sigma0", 0.1)

    es = SepCMAES(
        n=spec.n_total,
        sigma0=sigma0,
        x0=P.initial_theta(spec),
        popsize=popsize,
        seed=args.seed,
        maxiter=generations,
    )
    print(f"[train] sep-CMA-ES: λ={es.popsize}, σ0={sigma0}, m_cma={m_cma}, T={generations}, "
          f"budget≈{es.popsize * m_cma * generations}")

    run_dir = _REPO / "experiments" / args.benchmark / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    run_kwargs = dict(
        max_turns=args.max_turns or sess.get("max_turns", 5),
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        verifier_requires_prior_worker=sess.get("verifier_requires_prior_worker", True),
    )

    gen = 0
    while not es.stop() and gen < generations:
        t0 = time.time()
        thetas = es.ask()

        # Common random numbers: ALL candidates in a generation are scored on the
        # SAME minibatch (re-sampled across generations). This removes task-luck from
        # intra-generation ranking — the variance-reduction the noisy binary reward
        # needs for sep-CMA-ES to rank candidates by policy quality, not by which
        # tasks they happened to draw. (Pilot without this showed a flat/bouncing J.)
        gen_rng = random.Random(args.seed * 100000 + gen)
        gen_minibatch = sample_minibatch(tasks, m_cma, gen_rng)

        def minibatch_fn(i, _mb=gen_minibatch):
            return _mb

        def _on_cand(i, fit, elapsed, _g=gen):
            print(f"    [gen {_g} cand {i + 1}/{len(thetas)}] fit={fit:.3f} ({elapsed:.0f}s)",
                  flush=True)

        fits = await evaluate_population(
            thetas, spec, policy, pool, pool_models, minibatch_fn,
            sample=True, on_candidate=_on_cand, **run_kwargs
        )
        es.tell(thetas, fits)

        best_x, best_f = es.best()
        rec = {
            "generation": gen,
            "gen_mean_fitness": float(np.mean(fits)),
            "gen_max_fitness": float(np.max(fits)),
            "best_fitness": float(best_f),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"[gen {gen:3d}] mean={rec['gen_mean_fitness']:.3f} "
              f"max={rec['gen_max_fitness']:.3f} best={rec['best_fitness']:.3f} "
              f"({rec['seconds']}s)")

        np.save(run_dir / "best_theta.npy", best_x)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        gen += 1

    best_x, best_f = es.best()
    np.save(run_dir / "best_theta.npy", best_x)
    summary = {
        "benchmark": args.benchmark,
        "pool": pool_models,
        "n_total": spec.n_total,
        "popsize": es.popsize,
        "m_cma": m_cma,
        "generations": gen,
        "best_fitness": float(best_f),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] DONE. best_fitness={best_f:.4f}  -> {run_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve the TRINITY coordinator with sep-CMA-ES")
    ap.add_argument("--benchmark", required=True, help="math500 | mmlu | gpqa | livecodebench")
    ap.add_argument("--config", default=str(_REPO / "configs" / "trinity.yaml"))
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=256, dest="max_items")
    ap.add_argument("--max-turns", type=int, default=0, dest="max_turns", help="override K")
    ap.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    ap.add_argument("--reasoning", default="minimal")
    ap.add_argument("--generations", type=int, default=0, help="override config T")
    ap.add_argument("--popsize", type=int, default=0, help="override λ")
    ap.add_argument("--m-cma", type=int, default=0, dest="m_cma", help="override replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="run", dest="run_name")
    args = ap.parse_args()
    # argparse stores 0 for "not set" on the int overrides; normalize to None-ish.
    args.generations = args.generations or None
    args.popsize = args.popsize or None
    args.m_cma = args.m_cma or None
    args.max_turns = args.max_turns or None
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
