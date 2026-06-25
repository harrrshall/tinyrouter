"""Honest evaluation of a Conductor: PURE binary correctness, with cost.

Reports only :func:`trinity.fugu.reward.is_correct` (never the shaped training
reward), so a number here cannot be inflated by partial credit. Supports several
reps per task: the single-sample noise that, per docs/RESULTS.md, swung random
routing by about 6 points is denoised by averaging reps, and the per-query
binary it emits feeds straight into ``scripts/oracle_ceiling.py`` (which supplies
the winner's-curse-debiased routing ceiling and bootstrap CIs, the FP/FN-proof
verdict layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trinity.fugu.cost import CostMeter, price_table
from trinity.fugu.reward import is_correct
from trinity.fugu.workflow import propose_and_run
from trinity.types import Task

__all__ = ["EvalResult", "evaluate"]


@dataclass
class EvalResult:
    """Aggregate + per-task evaluation outcome with cost."""

    n_tasks: int
    reps: int
    accuracy: float                       # mean over tasks of (mean over reps)
    parse_rate: float                     # mean fraction of proposals that parsed
    per_task: dict = field(default_factory=dict)
    per_query_binary: dict = field(default_factory=dict)   # task_id -> 0/1 majority
    cost: dict = field(default_factory=dict)
    aborted: bool = False


async def evaluate(
    conductor,
    tasks: list[Task],
    pool,
    pool_models: list[str],
    *,
    reps: int = 1,
    max_depth: int = 1,
    temperature: float = 0.2,
    prices: dict | None = None,
    cap_usd: float = 0.0,
    client=None,
) -> EvalResult:
    """Evaluate ``conductor`` on ``tasks`` with ``reps`` samples each.

    Uses sampling when ``reps > 1`` (so the reps are independent draws), greedy
    otherwise. Respects a spend ``cap_usd`` (0 disables it): if the cap trips
    mid-run the result is returned with ``aborted=True`` and only the tasks
    completed so far, rather than overspending.
    """
    meter = CostMeter(prices=prices or price_table(), cap_usd=cap_usd)
    per_task: dict[str, dict] = {}
    per_query_binary: dict[str, int] = {}
    task_accs: list[float] = []
    parse_rates: list[float] = []
    aborted = False

    for task in tasks:
        votes: list[int] = []
        parsed: list[int] = []
        for _ in range(reps):
            run = await propose_and_run(
                conductor, task, pool, pool_models,
                sample=(reps > 1), max_depth=max_depth,
                temperature=temperature, reasoning="minimal", client=client,
            )
            meter.add_run(run)
            votes.append(is_correct(run, task))
            parsed.append(int(run.parsed_ok))
            if meter.aborted:
                aborted = True
                break
        if votes:
            acc = sum(votes) / len(votes)
            pr = sum(parsed) / len(parsed)
            per_task[task.task_id] = {
                "acc": acc, "reps_correct": votes, "parse_rate": pr,
            }
            # Majority vote (>= half) for the per-query 0/1 the diagnostic consumes.
            per_query_binary[task.task_id] = int(2 * sum(votes) >= len(votes))
            task_accs.append(acc)
            parse_rates.append(pr)
        if aborted:
            break

    accuracy = float(sum(task_accs) / len(task_accs)) if task_accs else 0.0
    parse_rate = float(sum(parse_rates) / len(parse_rates)) if parse_rates else 0.0
    return EvalResult(
        n_tasks=len(per_task),
        reps=reps,
        accuracy=accuracy,
        parse_rate=parse_rate,
        per_task=per_task,
        per_query_binary=per_query_binary,
        cost=meter.report(),
        aborted=aborted,
    )
