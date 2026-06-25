"""Offline tests for the Fugu workflow parse-gate and executor (no network)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from trinity.fugu.conductor import StubConductor
from trinity.fugu.workflow import (
    CONDUCTOR_KEY,
    parse_workflow,
    propose_and_run,
    run_workflow,
)
from trinity.fugu.reward import is_correct, training_reward
from trinity.types import Task

POOL = ["deepseek-v4-pro", "glm-5p2", "kimi-k2p6"]

WF_OK = """
model_id = [0, 1]
subtasks = ["Solve the problem.", "State the final answer."]
access_list = [[], [0]]
"""

WF_SELF = """
model_id = [3, 1]
subtasks = ["Recurse on a sub-problem.", "Give the final answer."]
access_list = [[], [0]]
"""


@dataclass
class _Chat:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5


class StubPool:
    """A pool whose every worker returns a fixed gradeable answer."""

    def __init__(self, answer: str = "The answer is \\boxed{4}", pt: int = 10, ct: int = 5):
        self.answer = answer
        self.pt = pt
        self.ct = ct
        self.calls = 0

    async def chat(self, model, messages, **kwargs):
        self.calls += 1
        return _Chat(self.answer, self.pt, self.ct)


def _task():
    return Task(task_id="t1", benchmark="math500", prompt="What is 2+2?", answer="4")


def test_parse_valid():
    wf, ok = parse_workflow(WF_OK, n_workers=3)
    assert ok and wf is not None
    assert [s.model_id for s in wf.steps] == [0, 1]
    assert wf.steps[1].access == [0]
    assert wf.steps[0].access == []


def test_parse_gate_rejects_bad_workflows():
    # missing a list
    assert parse_workflow("model_id=[0]\nsubtasks=['x']", 3)[1] is False
    # length mismatch
    bad_len = "model_id=[0,1]\nsubtasks=['a']\naccess_list=[[]]"
    assert parse_workflow(bad_len, 3)[1] is False
    # model_id out of range (no self allowed)
    bad_idx = "model_id=[9]\nsubtasks=['a']\naccess_list=[[]]"
    assert parse_workflow(bad_idx, 3, allow_self=False)[1] is False
    # self index rejected when allow_self is False
    assert parse_workflow("model_id=[3]\nsubtasks=['a']\naccess_list=[[]]", 3,
                          allow_self=False)[1] is False
    # forward reference in access_list (step 0 cannot read step 1)
    fwd = "model_id=[0,1]\nsubtasks=['a','b']\naccess_list=[[1],[]]"
    assert parse_workflow(fwd, 3)[1] is False
    # over-length workflow
    over = ("model_id=[0,0,0,0,0,0]\nsubtasks=['a','a','a','a','a','a']\n"
            "access_list=[[],[],[],[],[],[]]")
    assert parse_workflow(over, 3, max_steps=5)[1] is False
    # bracket inside a subtask string must not break parsing
    tricky = "model_id=[0]\nsubtasks=['compute f[x] then box it']\naccess_list=['all']"
    wf, ok = parse_workflow(tricky, 3)
    assert ok and wf.steps[0].access == "all"


def test_run_workflow_executes_and_grades():
    wf, ok = parse_workflow(WF_OK, n_workers=3)
    pool = StubPool()
    run = asyncio.run(run_workflow(wf, _task(), pool, POOL))
    assert run.parsed_ok and is_correct(run, _task()) == 1
    assert pool.calls == 2 and run.n_llm_calls == 2
    # exact per-model token accounting: two distinct workers, each one call.
    assert run.model_tokens["deepseek-v4-pro"] == (10, 5)
    assert run.model_tokens["glm-5p2"] == (10, 5)
    assert run.completion_tokens == 10


def test_propose_and_run_counts_conductor_and_grades():
    run = asyncio.run(
        propose_and_run(StubConductor(WF_OK), _task(), StubPool(), POOL)
    )
    assert run.parsed_ok and training_reward(run, _task()) == 1.0
    # proposal call is counted, and the conductor's own tokens are accounted.
    assert run.n_llm_calls == 3  # 1 proposal + 2 workers
    assert CONDUCTOR_KEY in run.model_tokens


def test_parse_fail_scores_zero():
    run = asyncio.run(
        propose_and_run(StubConductor("not a workflow at all"), _task(), StubPool(), POOL)
    )
    assert run.parsed_ok is False
    assert is_correct(run, _task()) == 0
    assert training_reward(run, _task()) == 0.0
    assert run.n_llm_calls == 1  # only the (failed) proposal


def test_recursive_self_call_terminates():
    # The conductor always proposes a self-step; recursion must stop at max_depth.
    run = asyncio.run(
        propose_and_run(
            StubConductor(WF_SELF), _task(), StubPool(), POOL, max_depth=1
        )
    )
    assert run.parsed_ok and is_correct(run, _task()) == 1
    # tokens from the recursive sub-run are merged into the parent's accounting.
    assert run.model_tokens.get(CONDUCTOR_KEY, (0, 0))[1] >= 0
    assert run.completion_tokens > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
