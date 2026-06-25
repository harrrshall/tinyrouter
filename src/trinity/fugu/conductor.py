"""The Conductor policy: propose a workflow for a query.

Two implementations:

* :class:`PromptedConductor` is a ZERO-TRAINING baseline. It asks a chat model
  (any Fireworks pool model, or a dedicated small model) to emit the three-list
  workflow. It establishes the "untrained Fugu-Ultra" reference the GRPO-trained
  Conductor must beat, and it is the cheapest way to exercise the full
  propose -> parse -> execute -> grade pipeline on real models.
* :class:`StubConductor` returns a fixed or callable-provided proposal for
  offline tests (no network).

The trained-LM backend (an HF model fine-tuned by GRPO on the remote H200) plugs
in by implementing the same :class:`Conductor` protocol; see
docs/fugu/REPLICATION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from trinity.fugu.workflow import MAX_STEPS
from trinity.types import Task

__all__ = ["Proposal", "Conductor", "PromptedConductor", "StubConductor", "build_prompt"]


@dataclass
class Proposal:
    """A Conductor's workflow proposal (raw text plus accounting)."""

    text: str
    completion_tokens: int = 0
    prompt_tokens: int = 0
    logprob: float | None = None     # filled by a trainable backend, else None


class Conductor(Protocol):
    """Anything that can propose a workflow for a task over a worker menu."""

    async def propose(
        self,
        task: Task,
        worker_names: list[str],
        *,
        sample: bool = False,
        rng=None,
        client=None,
    ) -> Proposal: ...


def _worker_menu(worker_names: list[str]) -> str:
    lines = [f"  {i}: {name}" for i, name in enumerate(worker_names)]
    lines.append(f"  {len(worker_names)}: yourself (recursive sub-workflow)")
    return "\n".join(lines)


def build_prompt(
    task: Task, worker_names: list[str], *, max_steps: int = MAX_STEPS
) -> list[dict]:
    """Chat messages instructing a model to emit a parseable workflow.

    The format described here is exactly what :func:`trinity.fugu.workflow.parse_workflow`
    accepts, so a well-behaved model's output passes the parse-gate.
    """
    n = len(worker_names)
    system = (
        "You are the Conductor: you orchestrate a pool of worker LLMs to solve a "
        "problem. You do NOT solve it yourself; you design a short workflow and "
        "assign each step to a worker.\n\n"
        "Available workers (by index):\n"
        f"{_worker_menu(worker_names)}\n\n"
        f"Design a workflow of 1 to {max_steps} steps. Output EXACTLY three Python "
        "lists of equal length and NOTHING else:\n"
        "  model_id    = [..]   # worker index for each step\n"
        "  subtasks    = [..]   # a clear instruction string for each step\n"
        "  access_list = [..]   # for each step: [] for none, \"all\", or a list of\n"
        "                       # earlier step indices whose outputs that step may read\n"
        "Rules: indices in access_list must be strictly smaller than the step's own "
        "index. The LAST step must produce the final answer in the required format. "
        f"Worker index {n} means call yourself recursively on a sub-problem."
    )
    user = f"Benchmark: {task.benchmark}\nProblem:\n{task.prompt}\n\nEmit the three lists."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class PromptedConductor:
    """Zero-training baseline Conductor backed by a chat model via the pool."""

    def __init__(self, pool, model: str, *, max_steps: int = MAX_STEPS,
                 sample_temperature: float = 0.7, greedy_temperature: float = 0.2,
                 max_tokens: int = 1024):
        self.pool = pool
        self.model = model
        self.max_steps = max_steps
        self.sample_temperature = sample_temperature
        self.greedy_temperature = greedy_temperature
        self.max_tokens = max_tokens

    async def propose(
        self, task: Task, worker_names: list[str], *,
        sample: bool = False, rng=None, client=None,
    ) -> Proposal:
        del rng  # temperature controls exploration for the prompted baseline
        messages = build_prompt(task, worker_names, max_steps=self.max_steps)
        temp = self.sample_temperature if sample else self.greedy_temperature
        kwargs = dict(temperature=temp, max_tokens=self.max_tokens)
        if client is not None:
            kwargs["client"] = client
        res = await self.pool.chat(self.model, messages, **kwargs)
        return Proposal(
            text=res.text,
            completion_tokens=getattr(res, "completion_tokens", 0),
            prompt_tokens=getattr(res, "prompt_tokens", 0),
        )


class StubConductor:
    """Offline Conductor for tests: a fixed string or a ``fn(task) -> str``."""

    def __init__(self, proposal: str | Callable[[Task], str]):
        self._proposal = proposal

    async def propose(
        self, task: Task, worker_names: list[str], *,
        sample: bool = False, rng=None, client=None,
    ) -> Proposal:
        text = self._proposal(task) if callable(self._proposal) else self._proposal
        return Proposal(text=text, completion_tokens=0)
