"""The Conductor's natural-language workflow: schema, parse-gate, executor.

Open replication of the Conductor (arXiv:2512.04388) / OpenFugu ``ultra``. The
Conductor proposes a workflow as three parallel lists:

    model_id    = [int, ...]   which worker runs each step (or SELF for recursion)
    subtasks    = [str, ...]   the natural-language instruction for each step
    access_list = [a, ...]     which prior step outputs feed each step:
                                  []      -> only the original query
                                  "all"   -> every strictly-prior step output
                                  [0, 2]  -> outputs of steps 0 and 2 only

The workflow is capped at :data:`MAX_STEPS`. Execution runs the steps in order;
the FINAL step's output is the synthesized answer (the terminal node reads prior
outputs via its access list, which is how the Conductor "synthesizes").

False-positive / false-negative posture
---------------------------------------
* The proposal text is NEVER executed. The three lists are parsed with
  ``ast.literal_eval`` only (a safe literal evaluator), and a strict parse-gate
  rejects any malformed, out-of-range, forward-referencing, or over-length
  workflow. A rejected workflow is reported as ``parsed_ok=False`` so the reward
  layer scores it 0, never a silent pass.
* Worker outputs are cleaned with the shared :mod:`trinity.roles.postprocess`,
  and each worker is told the exact answer format for its benchmark, so a
  correct answer is not lost to an unparseable shape (a false negative).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from trinity.roles import postprocess as _pp
from trinity.types import Role, Task

__all__ = [
    "MAX_STEPS",
    "WorkflowStep",
    "Workflow",
    "StepResult",
    "WorkflowRun",
    "parse_workflow",
    "run_workflow",
    "propose_and_run",
    "format_hint",
]

# The Conductor paper caps a workflow at 5 steps.
MAX_STEPS: int = 5

# Aliases accepted for each of the three lists (robust to minor wording drift).
_MODEL_KEYS = ("model_id", "model_ids", "models", "worker_id", "worker_ids", "worker")
_SUBTASK_KEYS = ("subtasks", "subtask", "tasks", "instructions", "steps_instructions")
_ACCESS_KEYS = ("access_list", "access_lists", "access", "context", "reads", "inputs")


@dataclass
class WorkflowStep:
    """One node of the workflow DAG."""

    model_id: int                 # worker index, or n_workers == SELF (recursion)
    subtask: str                  # natural-language instruction for this step
    access: object                # "all" | list[int] (strictly-prior step indices)


@dataclass
class Workflow:
    """A parsed, validated workflow (a DAG of up to MAX_STEPS steps)."""

    steps: list[WorkflowStep]


@dataclass
class StepResult:
    """The executed outcome of one workflow step."""

    step: int
    model_id: int
    model_name: str
    subtask: str
    output: str                   # post-processed O_k
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


# Key under which the Conductor's OWN generation tokens are accounted in
# ``WorkflowRun.model_tokens`` (distinct from the worker models). Cost of a run
# is conductor generation + every worker call, summed across recursion.
CONDUCTOR_KEY: str = "<conductor>"


@dataclass
class WorkflowRun:
    """The end-to-end result of proposing and executing a workflow.

    ``parsed_ok=False`` means the Conductor's proposal failed the parse-gate, so
    there is no final answer to grade and the training reward is the parse-fail
    floor (0.0). This is the single source of truth the reward layer reads.

    ``model_tokens`` maps each model name (and :data:`CONDUCTOR_KEY`) to a
    ``(prompt_tokens, completion_tokens)`` pair, summed over this run and every
    recursive sub-run, so :mod:`trinity.fugu.cost` can price the run exactly.
    """

    workflow: Workflow | None
    parsed_ok: bool
    steps: list[StepResult] = field(default_factory=list)
    final_answer: str = ""
    raw_proposal: str = ""
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_tokens: dict[str, tuple[int, int]] = field(default_factory=dict)


def _accum(acc: dict[str, tuple[int, int]], name: str, pt: int, ct: int) -> None:
    """Add ``(pt, ct)`` to the running per-model token totals."""
    p, c = acc.get(name, (0, 0))
    acc[name] = (p + pt, c + ct)


def _merge_tokens(dst: dict[str, tuple[int, int]], src: dict[str, tuple[int, int]]) -> None:
    """Fold a sub-run's per-model token totals into the parent's."""
    for name, (p, c) in src.items():
        _accum(dst, name, p, c)


# ---------------------------------------------------------------------------
# Parse-gate (the Conductor's format reward; r=0 on any failure here)
# ---------------------------------------------------------------------------
def _extract_bracketed(text: str, name: str) -> str | None:
    """Return the ``[...]`` literal assigned to ``name`` in ``text``, or ``None``.

    Scans for ``name`` followed by ``=`` or ``:`` then a balanced bracket span,
    honouring string literals so a ``]`` inside a subtask string does not close
    the list early. Does not evaluate anything (that is the caller's job, via a
    safe literal parser).
    """
    m = re.search(rf"(?<![\w]){re.escape(name)}\s*[:=]\s*", text)
    if not m:
        return None
    start = text.find("[", m.end())
    if start == -1:
        return None
    depth = 0
    in_str: str | None = None
    esc = False
    i = start
    while i < len(text):
        c = text[i]
        if in_str is not None:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
        else:
            if c in ("'", '"'):
                in_str = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _first_list(text: str, names: tuple[str, ...]) -> list | None:
    """First of ``names`` that parses to a Python list literal, else ``None``."""
    for name in names:
        span = _extract_bracketed(text, name)
        if span is None:
            continue
        try:
            val = ast.literal_eval(span)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(val, list):
            return val
    return None


def _normalize_access(acc: object, step_index: int) -> object | None:
    """Validate/normalize one access entry. Returns ``None`` if invalid.

    Valid forms: ``"all"``, ``[]``/``None`` (query only), or a list of integer
    indices that strictly precede ``step_index`` (a forward reference is an
    invalid DAG and rejects the whole workflow).
    """
    if acc is None:
        return []
    if isinstance(acc, str):
        return "all" if acc.strip().lower() == "all" else None
    if isinstance(acc, (list, tuple)):
        if len(acc) == 1 and isinstance(acc[0], str) and acc[0].strip().lower() == "all":
            return "all"
        out: list[int] = []
        for j in acc:
            if isinstance(j, bool) or not isinstance(j, int):
                return None
            if j < 0 or j >= step_index:
                return None
            out.append(j)
        return sorted(set(out))
    return None


def parse_workflow(
    text: str,
    n_workers: int,
    *,
    max_steps: int = MAX_STEPS,
    allow_self: bool = True,
) -> tuple[Workflow | None, bool]:
    """Parse a Conductor proposal into a validated :class:`Workflow`.

    Returns ``(workflow, True)`` on success or ``(None, False)`` if the proposal
    fails any gate: missing/extra lists, length mismatch, empty or over-length
    workflow, non-int or out-of-range ``model_id``, empty subtask, or an invalid
    access entry. ``model_id == n_workers`` denotes a recursive self-call and is
    accepted only when ``allow_self`` is set.

    Args:
        text: the Conductor's raw proposal text.
        n_workers: number of worker models in the pool.
        max_steps: maximum allowed workflow length.
        allow_self: whether a self-call worker index (``n_workers``) is valid.

    Returns:
        ``(Workflow, True)`` or ``(None, False)``.
    """
    if not text or n_workers < 1:
        return None, False
    models = _first_list(text, _MODEL_KEYS)
    subtasks = _first_list(text, _SUBTASK_KEYS)
    access = _first_list(text, _ACCESS_KEYS)
    if models is None or subtasks is None or access is None:
        return None, False
    n = len(models)
    if n == 0 or n > max_steps:
        return None, False
    if not (len(subtasks) == n and len(access) == n):
        return None, False

    hi = n_workers if allow_self else n_workers - 1
    steps: list[WorkflowStep] = []
    for i in range(n):
        mid = models[i]
        if isinstance(mid, bool) or not isinstance(mid, int):
            return None, False
        if mid < 0 or mid > hi:
            return None, False
        sub = subtasks[i]
        if not isinstance(sub, str) or not sub.strip():
            return None, False
        norm = _normalize_access(access[i], i)
        if norm is None:
            return None, False
        steps.append(WorkflowStep(model_id=mid, subtask=sub.strip(), access=norm))
    return Workflow(steps=steps), True


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def format_hint(benchmark: str) -> str:
    """Answer-format instruction for a worker, to keep outputs gradeable.

    Lowering false negatives: a correct answer in an unparseable shape scores 0,
    so every worker is told exactly how to present the final answer for this
    benchmark, matching what :mod:`trinity.orchestration.reward` extracts.
    """
    key = (benchmark or "").strip().lower()
    if key in {"mmlu", "gpqa", "gpqa-diamond", "gpqa_diamond"}:
        return "End with the final answer on its own line as: Answer: X  (X is one of A, B, C, D)."
    if key in {"livecodebench", "lcb", "bigcodebench", "bigcode"}:
        return "Provide the complete solution as a single Python code block in triple backticks."
    # math500 / aime and any default.
    return "End with the final answer on its own line as \\boxed{ANSWER}."


def _context_block(access: object, outputs: list[str]) -> str:
    """Render the prior-step outputs this step is allowed to read."""
    if access == "all":
        chosen = list(enumerate(outputs))
    elif isinstance(access, list) and access:
        chosen = [(j, outputs[j]) for j in access if 0 <= j < len(outputs)]
    else:
        chosen = []
    if not chosen:
        return ""
    parts = [f"[Output of step {j}]\n{txt}" for j, txt in chosen]
    return "Context from earlier steps:\n" + "\n\n".join(parts) + "\n\n"


def _worker_messages(task: Task, subtask: str, context: str) -> list[dict]:
    """Build the chat messages for one worker step."""
    system = (
        "You are an expert worker model in a multi-agent system. Carry out the "
        "assigned subtask precisely and concisely."
    )
    user = (
        f"Original problem:\n{task.prompt}\n\n"
        f"{context}"
        f"Your subtask:\n{subtask}\n\n"
        f"{format_hint(task.benchmark)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _supported_kwargs(fn, kwargs: dict) -> dict:
    """Drop kwargs a (possibly stub) ``chat`` does not accept (cf. session.py)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


async def _worker_call(
    pool,
    model_name: str,
    task: Task,
    subtask: str,
    context: str,
    *,
    temperature: float,
    max_tokens: int,
    reasoning: str | None,
    client,
) -> tuple[str, int, int]:
    """Call one worker; return ``(raw_text, prompt_tokens, completion_tokens)``."""
    messages = _worker_messages(task, subtask, context)
    kwargs = dict(temperature=temperature, max_tokens=max_tokens)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    if client is not None:
        kwargs["client"] = client
    res = await pool.chat(model_name, messages, **_supported_kwargs(pool.chat, kwargs))
    return (
        res.text,
        getattr(res, "prompt_tokens", 0),
        getattr(res, "completion_tokens", 0),
    )


def _subtask_as_task(task: Task, subtask: str, context: str) -> Task:
    """Wrap a recursive self-call's subtask+context as a fresh Task for the inner run."""
    prompt = f"{task.prompt}\n\n{context}Focus on this subtask:\n{subtask}".strip()
    return Task(
        task_id=f"{task.task_id}::self",
        benchmark=task.benchmark,
        prompt=prompt,
        answer=task.answer,
        meta=dict(task.meta),
    )


async def run_workflow(
    workflow: Workflow,
    task: Task,
    pool,
    pool_models: list[str],
    *,
    conductor=None,
    max_depth: int = 1,
    depth: int = 0,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    reasoning: str | None = "minimal",
    client=None,
) -> WorkflowRun:
    """Execute a validated workflow over the pool and return the run record.

    Steps run in order. Each step sees the original query plus the prior outputs
    its access list selects. ``model_id == len(pool_models)`` is a recursive
    self-call: if recursion budget remains (``depth < max_depth``) and a
    ``conductor`` is supplied, the conductor proposes a sub-workflow for the
    step's subtask and the sub-run's final answer becomes the step output;
    otherwise the step degrades gracefully to worker 0 so the workflow still
    produces an answer (never a crash, never a silent empty pass).

    The final step's output is the synthesized answer.
    """
    n_workers = len(pool_models)
    self_id = n_workers
    steps_out: list[StepResult] = []
    outputs: list[str] = []
    n_calls = 0
    ptoks = 0
    ctoks = 0
    model_tokens: dict[str, tuple[int, int]] = {}

    for i, st in enumerate(workflow.steps):
        context = _context_block(st.access, outputs)
        is_self = st.model_id == self_id
        can_recurse = is_self and conductor is not None and depth < max_depth

        if can_recurse:
            sub_task = _subtask_as_task(task, st.subtask, context)
            prop = await conductor.propose(
                sub_task, pool_models, sample=False, client=client
            )
            sub_wf, ok = parse_workflow(prop.text, n_workers, allow_self=True)
            p_pt = getattr(prop, "prompt_tokens", 0)
            p_ct = getattr(prop, "completion_tokens", 0)
            _accum(model_tokens, CONDUCTOR_KEY, p_pt, p_ct)
            ptoks += p_pt
            ctoks += p_ct
            n_calls += 1  # the recursive proposal call
            if ok:
                sub_run = await run_workflow(
                    sub_wf,
                    sub_task,
                    pool,
                    pool_models,
                    conductor=conductor,
                    max_depth=max_depth,
                    depth=depth + 1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    client=client,
                )
                # The sub-run's per-model tokens already cover its workers and any
                # deeper conductor calls; merge them (do NOT re-accumulate below).
                _merge_tokens(model_tokens, sub_run.model_tokens)
                ptoks += sub_run.prompt_tokens
                ctoks += sub_run.completion_tokens
                n_calls += sub_run.n_llm_calls
                out = sub_run.final_answer
                raw = out
                mname = "self"
                pt, ct = sub_run.prompt_tokens, sub_run.completion_tokens
            else:
                # Sub-proposal failed the parse-gate: fall back to worker 0.
                mname = pool_models[0]
                raw, pt, ct = await _worker_call(
                    pool, mname, task, st.subtask, context,
                    temperature=temperature, max_tokens=max_tokens,
                    reasoning=reasoning, client=client,
                )
                out = _pp.postprocess(raw, Role.WORKER)
                _accum(model_tokens, mname, pt, ct)
                ptoks += pt
                ctoks += ct
                n_calls += 1
        else:
            # Plain worker step (a self-call with no recursion budget maps to 0).
            mname = pool_models[0] if is_self else pool_models[st.model_id]
            raw, pt, ct = await _worker_call(
                pool, mname, task, st.subtask, context,
                temperature=temperature, max_tokens=max_tokens,
                reasoning=reasoning, client=client,
            )
            out = _pp.postprocess(raw, Role.WORKER)
            _accum(model_tokens, mname, pt, ct)
            ptoks += pt
            ctoks += ct
            n_calls += 1

        steps_out.append(
            StepResult(
                step=i, model_id=st.model_id, model_name=mname, subtask=st.subtask,
                output=out, raw=raw, prompt_tokens=pt, completion_tokens=ct,
            )
        )
        outputs.append(out)

    final = outputs[-1] if outputs else ""
    return WorkflowRun(
        workflow=workflow, parsed_ok=True, steps=steps_out, final_answer=final,
        raw_proposal="", n_llm_calls=n_calls, prompt_tokens=ptoks,
        completion_tokens=ctoks, model_tokens=model_tokens,
    )


async def propose_and_run(
    conductor,
    task: Task,
    pool,
    pool_models: list[str],
    *,
    sample: bool = False,
    rng=None,
    max_depth: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    reasoning: str | None = "minimal",
    client=None,
) -> WorkflowRun:
    """Full pipeline: conductor proposes, parse-gate, then execute (or score 0).

    On a parse-gate failure returns a ``WorkflowRun`` with ``parsed_ok=False``
    (no steps, empty answer) so the reward layer scores it the parse-fail floor.
    """
    prop = await conductor.propose(
        task, pool_models, sample=sample, rng=rng, client=client
    )
    wf, ok = parse_workflow(
        prop.text, len(pool_models), allow_self=(max_depth > 0)
    )
    p_pt = getattr(prop, "prompt_tokens", 0)
    p_ct = getattr(prop, "completion_tokens", 0)
    if not ok:
        return WorkflowRun(
            workflow=None, parsed_ok=False, steps=[], final_answer="",
            raw_proposal=prop.text, n_llm_calls=1,
            prompt_tokens=p_pt, completion_tokens=p_ct,
            model_tokens={CONDUCTOR_KEY: (p_pt, p_ct)},
        )
    run = await run_workflow(
        wf, task, pool, pool_models, conductor=conductor, max_depth=max_depth,
        temperature=temperature, max_tokens=max_tokens, reasoning=reasoning,
        client=client,
    )
    run.raw_proposal = prop.text
    run.n_llm_calls += 1  # count the top-level proposal call
    run.prompt_tokens += p_pt
    run.completion_tokens += p_ct
    _accum(run.model_tokens, CONDUCTOR_KEY, p_pt, p_ct)
    return run
