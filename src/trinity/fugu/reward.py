"""The Conductor reward, built on the project's FIXED grader.

Two functions, kept deliberately separate to avoid false positives/negatives:

* :func:`training_reward` is the GRPO signal. It is the Conductor paper's
  two-stage reward: 0.0 if the proposal failed the parse-gate (format), else
  1.0 if the synthesized answer is correct and 0.5 otherwise (partial credit so
  a parseable-but-wrong workflow still ranks above a malformed one).
* :func:`is_correct` is the EVALUATION signal: a PURE 0/1 with no partial
  credit and no format leniency. Reporting must use this, never the shaped
  training reward, exactly as the repo keeps shaped training fitness separate
  from pure-binary eval (see docs/RESULTS.md and optim/fitness.py).

Correctness is decided ONLY by :func:`trinity.orchestration.reward.score_text`,
the same de-bugged extractor used by ``eval.py`` and the oracle-ceiling
diagnostic. We do not reimplement extraction here; that is what stops a new
grading path from drifting into false positives or negatives.
"""
from __future__ import annotations

from trinity.orchestration import reward as _R

__all__ = ["committed_answer", "is_correct", "training_reward", "PARSE_FAIL", "PARTIAL"]

PARSE_FAIL: float = 0.0   # workflow did not parse (format gate)
PARTIAL: float = 0.5      # parsed and ran, but the answer is wrong


def committed_answer(benchmark: str, run) -> str:
    """Pick the text to grade from a workflow run.

    Mirrors :func:`trinity.orchestration.reward._committed_answer`: prefer the
    final (synthesized) answer, but if it has no extractable answer of the right
    shape, fall back to the most recent step output that does. This recovers
    answers the system actually produced and avoids a false negative when the
    terminal step rephrased without re-boxing the result. It NEVER changes which
    answer is judged correct, only which text the fixed extractor reads.
    """
    final = run.final_answer or ""
    if _R.has_answer(benchmark, final):
        return final
    for step in reversed(getattr(run, "steps", []) or []):
        txt = getattr(step, "output", "") or ""
        if _R.has_answer(benchmark, txt):
            return txt
    return final


def is_correct(run, task) -> int:
    """PURE 0/1 correctness of a workflow run (the only number eval may report).

    A run that failed the parse-gate is incorrect by definition. Otherwise the
    committed answer is graded by the shared fixed grader.
    """
    if not getattr(run, "parsed_ok", False):
        return 0
    candidate = committed_answer(task.benchmark, run)
    return int(_R.score_text(task.benchmark, candidate, task.answer) >= 1.0)


def training_reward(
    run, task, *, partial: float = PARTIAL, parse_fail: float = PARSE_FAIL
) -> float:
    """Two-stage GRPO reward: parse-gate, then correctness (1.0 / partial).

    Returns ``parse_fail`` if the workflow did not parse, ``1.0`` if the answer
    is correct, else ``partial``. Use this ONLY for training; report
    :func:`is_correct`.
    """
    if not getattr(run, "parsed_ok", False):
        return float(parse_fail)
    return 1.0 if is_correct(run, task) else float(partial)
