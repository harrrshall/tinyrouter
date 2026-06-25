"""Offline tests for GRPO math, rollout collection, the loop, and the cost cap."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from trinity.fugu.conductor import Proposal, StubConductor
from trinity.fugu.cost import CostMeter, price_table
from trinity.fugu.grpo import (
    GRPOConfig,
    GroupResult,
    collect_group,
    group_advantages,
    train,
)
from trinity.types import Task

POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]
WF_OK = "model_id=[0,1]\nsubtasks=['solve','answer']\naccess_list=[[],[0]]"
WF_BAD = "I cannot help with that."


@dataclass
class _Chat:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5


class StubPool:
    def __init__(self, answer="\\boxed{4}", ct=5):
        self.answer = answer
        self.ct = ct

    async def chat(self, model, messages, **kwargs):
        return _Chat(self.answer, 10, self.ct)


class CyclingConductor:
    """Alternates good and malformed proposals so rewards (and advantages) vary."""

    def __init__(self, texts):
        self.texts = texts
        self.i = 0

    async def propose(self, task, worker_names, *, sample=False, rng=None, client=None):
        text = self.texts[self.i % len(self.texts)]
        self.i += 1
        return Proposal(text=text, completion_tokens=3)


def _task():
    return Task(task_id="t", benchmark="math500", prompt="2+2", answer="4")


def test_group_advantages_math():
    assert group_advantages([]) == []
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]  # no signal
    adv = group_advantages([1.0, 0.0])
    assert adv[0] > 0 > adv[1]
    assert abs(sum(adv)) < 1e-9  # mean-centered


def test_collect_group_scores_and_advantages():
    cfg = GRPOConfig(group_size=4)
    conductor = CyclingConductor([WF_OK, WF_BAD])
    meter = CostMeter()
    g = asyncio.run(
        collect_group(conductor, _task(), StubPool(), POOL, cfg, meter=meter)
    )
    assert len(g.rewards) == 4 and len(g.advantages) == 4
    # two good (reward 1.0) and two parse-fail (reward 0.0) by construction.
    assert g.n_parsed == 2 and g.n_correct == 2
    assert sorted(g.rewards) == [0.0, 0.0, 1.0, 1.0]
    assert meter.spend > 0  # worker calls cost money; conductor is local (free)


class StubBackend:
    """A PolicyBackend that proposes via a StubConductor and counts updates."""

    def __init__(self, text):
        self._c = StubConductor(text)
        self.updates = 0

    async def propose(self, task, worker_names, *, sample=False, rng=None, client=None):
        return await self._c.propose(task, worker_names, sample=sample, rng=rng, client=client)

    def update(self, groups):
        self.updates += 1
        return {"n_groups": len(groups)}


def test_train_loop_runs_and_reports_cost():
    cfg = GRPOConfig(group_size=2, iterations=3, questions_per_iter=1)
    backend = StubBackend(WF_OK)
    out = asyncio.run(train(backend, [_task()], StubPool(), POOL, cfg))
    assert out["iterations"] == 3
    assert backend.updates == 3
    assert out["cost"]["spend_usd"] > 0
    assert out["final_accuracy"] == 1.0


def test_cost_cap_aborts_training():
    # Tiny cap with non-trivial worker tokens: the loop must stop early, not overspend.
    cfg = GRPOConfig(group_size=4, iterations=50, questions_per_iter=1, max_cost_usd=0.001)
    backend = StubBackend(WF_OK)
    out = asyncio.run(train(backend, [_task()], StubPool(ct=2000), POOL, cfg))
    assert out["cost"]["aborted"] is True
    assert out["iterations"] < 50


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
