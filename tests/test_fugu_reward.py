"""Offline tests for the Conductor reward: the false-positive/negative guards."""
from __future__ import annotations

from trinity.fugu.reward import committed_answer, is_correct, training_reward
from trinity.fugu.workflow import StepResult, WorkflowRun
from trinity.types import Task


def _run(final, *, parsed=True, steps=None):
    return WorkflowRun(
        workflow=None, parsed_ok=parsed, steps=steps or [], final_answer=final,
    )


def _step(out):
    return StepResult(step=0, model_id=0, model_name="glm-5p2", subtask="", output=out)


MATH = Task(task_id="m", benchmark="math500", prompt="2+2", answer="4")
MMLU = Task(task_id="c", benchmark="mmlu", prompt="pick", answer="C")


def test_parse_fail_is_incorrect_and_zero_reward():
    run = _run("\\boxed{4}", parsed=False)
    assert is_correct(run, MATH) == 0
    assert training_reward(run, MATH) == 0.0


def test_correct_and_wrong():
    assert is_correct(_run("\\boxed{4}"), MATH) == 1
    assert training_reward(_run("\\boxed{4}"), MATH) == 1.0
    assert is_correct(_run("\\boxed{5}"), MATH) == 0
    # parsed but wrong gets partial credit in TRAINING only.
    assert training_reward(_run("\\boxed{5}"), MATH) == 0.5


def test_committed_answer_recovers_from_a_non_boxed_final_step():
    # Final step rephrased without re-boxing; an earlier step had the boxed answer.
    run = _run("So the result follows.", steps=[_step("\\boxed{4}")])
    assert committed_answer("math500", run) == "\\boxed{4}"
    assert is_correct(run, MATH) == 1  # not a false negative


def test_no_false_positive_from_prose_letter():
    # "A nice approach" must NOT be read as choice 'A' (the JOURNAL P2 bug).
    run = _run("A nice approach to this question.")
    assert is_correct(run, MMLU) == 0
    # but an explicit answer line is graded correctly.
    assert is_correct(_run("A nice approach, but the answer is C."), MMLU) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
