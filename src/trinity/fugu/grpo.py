"""GRPO for the Conductor: group-normalized advantages, no KL, with cost guards.

This is the open replication of the Conductor's training (arXiv:2512.04388):
GRPO with group size G, advantage = group-normalized reward, and KL beta = 0.
The pieces here are framework-agnostic and offline-testable:

* :func:`group_advantages` is the pure GRPO advantage (mean-subtracted,
  std-normalized within a group). Unit-tested, no I/O.
* :func:`collect_group` does one group's rollouts: it samples G workflows from
  the Conductor, executes them over the pool, scores each with the TWO-STAGE
  training reward, meters the cost, and returns rewards + advantages.
* :func:`train` is the loop skeleton. It drives rollouts and hands each group's
  ``(samples, advantages)`` to a :class:`PolicyBackend`. The backend is where the
  actual gradient step lives; on the remote H200 that is an HF/TRL model, and for
  tests it is a stub. We keep the policy-gradient update behind this seam so the
  loop, the reward, and the cost accounting are all testable with zero GPU.

The reward used for training is :func:`trinity.fugu.reward.training_reward`
(parse-gate + 1.0/0.5). Reporting must use :func:`trinity.fugu.reward.is_correct`
(pure binary). Keeping those apart is how we avoid optimizing a metric that would
read as a false win.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trinity.fugu.cost import CostMeter, price_table
from trinity.fugu.reward import is_correct, training_reward
from trinity.fugu.workflow import propose_and_run
from trinity.types import Task

__all__ = [
    "GRPOConfig",
    "group_advantages",
    "collect_group",
    "PolicyBackend",
    "train",
    "CostCapExceeded",
]


class CostCapExceeded(RuntimeError):
    """Raised when a training/eval loop crosses its configured spend cap."""


@dataclass
class GRPOConfig:
    """GRPO hyperparameters (defaults match the Conductor paper)."""

    group_size: int = 64              # G rollouts per question
    iterations: int = 200
    questions_per_iter: int = 4       # batch = group_size * questions_per_iter
    lr: float = 1e-6
    kl_beta: float = 0.0              # Conductor uses NO KL penalty
    sample_temperature: float = 1.0   # train-time exploration temperature
    max_depth: int = 1                # recursive self-call budget
    partial_reward: float = 0.5
    max_cost_usd: float = 0.0         # 0 disables the cap; >0 aborts the run


def group_advantages(rewards: list[float], *, eps: float = 1e-6) -> list[float]:
    """GRPO advantage: ``(r - mean) / std`` within the group.

    A group whose rewards are all equal (std ~ 0) carries no learning signal, so
    every advantage is 0 (no spurious gradient from numerical noise).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


@dataclass
class GroupResult:
    """One question's group rollout: runs, rewards, advantages, and stats."""

    task: Task
    runs: list = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    n_parsed: int = 0
    n_correct: int = 0


async def collect_group(
    conductor,
    task: Task,
    pool,
    pool_models: list[str],
    cfg: GRPOConfig,
    *,
    meter: CostMeter | None = None,
    rng=None,
    client=None,
) -> GroupResult:
    """Sample ``cfg.group_size`` workflows for ``task``, run, score, and meter.

    Stops early and raises :class:`CostCapExceeded` if the meter crosses its cap
    mid-group, so a runaway training job cannot silently overspend.
    """
    runs: list = []
    rewards: list[float] = []
    for _ in range(cfg.group_size):
        run = await propose_and_run(
            conductor, task, pool, pool_models,
            sample=True, rng=rng, max_depth=cfg.max_depth,
            temperature=0.2, reasoning="minimal", client=client,
        )
        if meter is not None:
            meter.add_run(run)
        runs.append(run)
        rewards.append(training_reward(run, task, partial=cfg.partial_reward))
        if meter is not None and meter.aborted:
            raise CostCapExceeded(
                f"spend ${meter.spend:.2f} exceeded cap ${meter.cap_usd:.2f}"
            )
    return GroupResult(
        task=task,
        runs=runs,
        rewards=rewards,
        advantages=group_advantages(rewards),
        n_parsed=sum(1 for r in runs if r.parsed_ok),
        n_correct=sum(is_correct(r, task) for r in runs),
    )


class PolicyBackend(Protocol):
    """The trainable Conductor policy (HF model on the remote box; stub in tests).

    It both PROPOSES workflows (so it doubles as the ``Conductor`` for rollouts)
    and consumes ``(GroupResult)`` batches to take a gradient step.
    """

    async def propose(self, task: Task, worker_names: list[str], *,
                      sample: bool = False, rng=None, client=None): ...

    def update(self, groups: list[GroupResult]) -> dict:
        """Apply one GRPO update from a batch of groups; return train stats."""
        ...


async def train(
    backend: PolicyBackend,
    tasks: list[Task],
    pool,
    pool_models: list[str],
    cfg: GRPOConfig,
    *,
    prices: dict | None = None,
    rng=None,
    client=None,
    on_iter=None,
) -> dict:
    """GRPO training loop skeleton. Returns a summary including total cost.

    Each iteration samples ``cfg.questions_per_iter`` tasks (round-robin over the
    provided ``tasks``), collects a group per task, and calls ``backend.update``.
    The real gradient step lives in ``backend``; everything else (rollouts,
    reward, cost) runs here and is offline-testable with a stub backend and a
    stub pool.
    """
    meter = CostMeter(prices=prices or price_table(), cap_usd=cfg.max_cost_usd)
    history: list[dict] = []
    n = len(tasks)
    if n == 0:
        return {"iterations": 0, "cost": meter.report(), "history": history}

    cursor = 0
    for it in range(cfg.iterations):
        batch = [tasks[(cursor + j) % n] for j in range(cfg.questions_per_iter)]
        cursor = (cursor + cfg.questions_per_iter) % n
        groups: list[GroupResult] = []
        try:
            for task in batch:
                groups.append(
                    await collect_group(
                        backend, task, pool, pool_models, cfg,
                        meter=meter, rng=rng, client=client,
                    )
                )
        except CostCapExceeded:
            history.append({"iter": it, "aborted_cost": True})
            break

        stats = backend.update(groups)
        rec = {
            "iter": it,
            "mean_reward": _safe_mean([r for g in groups for r in g.rewards]),
            "accuracy": _safe_mean(
                [g.n_correct / max(1, len(g.runs)) for g in groups]
            ),
            "parse_rate": _safe_mean(
                [g.n_parsed / max(1, len(g.runs)) for g in groups]
            ),
            "spend_usd": round(meter.spend, 4),
            "update": stats,
        }
        history.append(rec)
        if on_iter is not None:
            on_iter(rec)
        if meter.aborted:
            break

    # Last completed iteration's accuracy (skip a trailing cost-abort record,
    # which carries no accuracy).
    final_accuracy = 0.0
    for rec in reversed(history):
        if "accuracy" in rec:
            final_accuracy = rec["accuracy"]
            break

    return {
        "iterations": len(history),
        "cost": meter.report(),
        "history": history,
        "final_accuracy": final_accuracy,
    }


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0
