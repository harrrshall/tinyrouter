#!/usr/bin/env python3
"""Train the Fugu Conductor backend with GRPO.

Two modes:

* ``--stub-pool``: no Fireworks calls and no spend. The local HF Conductor still
  loads, samples workflows, and takes GRPO updates against a deterministic fake
  worker. Use this first on the GPU box to validate CUDA/model plumbing.
* default: paid Fireworks worker rollouts. Requires ``FIREWORKS_API_KEY`` and a
  conservative ``--max-cost-usd`` cap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load_tasks(args):
    from trinity.types import Task

    if args.tasks_json:
        data = json.loads(Path(args.tasks_json).read_text())
        tasks = [
            Task(
                task_id=t["task_id"],
                benchmark=t["benchmark"],
                prompt=t["prompt"],
                answer=t["answer"],
                meta=t.get("meta", {}),
            )
            for t in data
        ]
    else:
        from trinity.orchestration.dataset import load_tasks

        tasks = load_tasks(args.benchmark, args.split, max_items=None, seed=args.seed)
    return tasks[: args.max_items] if args.max_items else tasks


@dataclass
class _Chat:
    text: str
    prompt_tokens: int = 64
    completion_tokens: int = 16


class StubPool:
    """No-spend worker pool for CUDA/backend smoke tests."""

    models = {
        "deepseek-v4-pro": "stub/deepseek-v4-pro",
        "glm-5p2": "stub/glm-5p2",
        "kimi-k2p6": "stub/kimi-k2p6",
    }

    async def chat(self, model, messages, **kwargs):
        del model, messages, kwargs
        return _Chat("\\boxed{4}")


async def _run(args) -> int:
    from trinity.fugu.cost import price_table
    from trinity.fugu.grpo import GRPOConfig, train
    from trinity.fugu.hf_backend import HFBackendConfig, HFPolicyBackend
    from trinity.fugu.workflow import CONDUCTOR_KEY

    tasks = _load_tasks(args)
    if not tasks:
        raise SystemExit("no tasks loaded")

    if args.stub_pool:
        pool = StubPool()
        prices = {name: (0.0, 0.0) for name in pool.models}
        prices[CONDUCTOR_KEY] = (0.0, 0.0)
    else:
        from trinity.llm.fireworks_client import FireworksPool

        pool = FireworksPool(args.models)
        prices = price_table(conductor_local=True)

    pool_models = list(pool.models)
    backend = HFPolicyBackend(
        HFBackendConfig(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            sample_temperature=args.sample_temperature,
            gradient_accumulation=args.gradient_accumulation,
            proposal_prefix=args.proposal_prefix,
        ),
        worker_names=pool_models,
    )
    cfg = GRPOConfig(
        group_size=args.group_size,
        iterations=args.iterations,
        questions_per_iter=args.questions_per_iter,
        lr=args.lr,
        sample_temperature=args.sample_temperature,
        max_depth=args.max_depth,
        max_cost_usd=args.max_cost_usd,
    )

    print(
        json.dumps(
            {
                "mode": "stub" if args.stub_pool else "fireworks",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device": args.device,
                "model": args.model_name,
                "tasks": len(tasks),
                "pool": pool_models,
                "group_size": cfg.group_size,
                "iterations": cfg.iterations,
                "questions_per_iter": cfg.questions_per_iter,
                "max_depth": cfg.max_depth,
                "max_cost_usd": cfg.max_cost_usd,
                "proposal_prefix": args.proposal_prefix,
            },
            indent=2,
        ),
        flush=True,
    )

    def _on_iter(rec):
        print("[iter] " + json.dumps(rec, default=str), flush=True)

    out = await train(
        backend,
        tasks,
        pool,
        pool_models,
        cfg,
        prices=prices,
        on_iter=_on_iter,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(out, indent=2, default=str))
    if args.save_model:
        backend.save_pretrained(str(out_dir / args.save_model))

    print("[summary] " + json.dumps(out, default=str), flush=True)
    print(f"[train] wrote {summary_path}", flush=True)
    return 2 if out["cost"].get("aborted") else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO train the Fugu HF Conductor")
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tasks-json", default="", dest="tasks_json")
    ap.add_argument("--models", default=str(_REPO / "configs" / "models.yaml"))
    ap.add_argument("--max-items", type=int, default=4, dest="max_items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B", dest="model_name")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--group-size", type=int, default=4, dest="group_size")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--questions-per-iter", type=int, default=1, dest="questions_per_iter")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--sample-temperature", type=float, default=1.0, dest="sample_temperature")
    ap.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    ap.add_argument("--max-prompt-tokens", type=int, default=4096, dest="max_prompt_tokens")
    ap.add_argument("--gradient-accumulation", type=int, default=1, dest="gradient_accumulation")
    ap.add_argument("--proposal-prefix", default="model_id = [", dest="proposal_prefix")
    ap.add_argument("--max-depth", type=int, default=0, dest="max_depth")
    ap.add_argument("--max-cost-usd", type=float, default=0.0, dest="max_cost_usd")
    ap.add_argument("--stub-pool", action="store_true", help="use fake workers; no API spend")
    ap.add_argument("--out-dir", default=str(_REPO / "experiments" / "fugu_grpo"), dest="out_dir")
    ap.add_argument("--summary-name", default="summary.json", dest="summary_name")
    ap.add_argument("--save-model", default="", dest="save_model")
    args = ap.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
