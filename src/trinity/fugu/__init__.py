"""Fugu-Ultra: an open replication of the Conductor (arXiv:2512.04388).

This package adds the *orchestration* half that TinyRouter's TRINITY router
lacks, mirroring OpenFugu (trotsky1997/OpenFugu) but built natively on this
repo's pieces so the grader is the same FIXED, de-bugged grader the rest of the
project uses (``trinity.orchestration.reward``). That reuse is deliberate: it is
how we avoid reintroducing the false positives (prose ``"A"`` read as a choice)
and false negatives (LiveCodeBench reward stuck at 0) that the JOURNAL records
fixing.

Layout (mirrors ``openfugu/`` upstream):

* :mod:`trinity.fugu.workflow`   the natural-language workflow schema, a strict
  parse-gate, and the executor (access-list topology + bounded recursive
  self-call). This is the "Fugu-Ultra" engine.
* :mod:`trinity.fugu.reward`     the two-stage Conductor reward (parse-gate then
  correctness) built ON TOP of the shared grader, plus a PURE-BINARY
  ``is_correct`` used for honest evaluation.
* :mod:`trinity.fugu.conductor`  the policy that proposes a workflow: a prompted
  baseline (zero training) and a stub for offline tests; the trained-LM backend
  is wired on the remote box (see docs/fugu/REPLICATION_PLAN.md).
* :mod:`trinity.fugu.grpo`       framework-agnostic GRPO math (group-normalized
  advantages, no KL) and the rollout/loop skeleton.

The worker pool is the existing open-source Fireworks trio: deepseek-v4-pro,
glm-5p2, kimi-k2p6.
"""
from __future__ import annotations

from trinity.fugu.workflow import (
    MAX_STEPS,
    StepResult,
    Workflow,
    WorkflowRun,
    WorkflowStep,
    parse_workflow,
    propose_and_run,
    run_workflow,
)
from trinity.fugu.reward import committed_answer, is_correct, training_reward

__all__ = [
    "MAX_STEPS",
    "Workflow",
    "WorkflowStep",
    "WorkflowRun",
    "StepResult",
    "parse_workflow",
    "run_workflow",
    "propose_and_run",
    "training_reward",
    "is_correct",
    "committed_answer",
]
