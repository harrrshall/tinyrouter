"""Per-task binary reward checkers for TRINITY trajectories.

The reward is the fitness signal sep-CMA-ES optimizes (SPEC §0.3.6, §5.2): a
single terminal Bernoulli ``R(tau) in {0, 1}`` per atomic evaluation. This
module is the single source of truth for "is this trajectory's final answer
correct?", dispatched on ``Trajectory.task.benchmark``.

Supported benchmarks
--------------------
* ``math500`` / ``aime``
    Extract a ``\\boxed{...}`` answer (else the last number) from the final
    answer, normalize, and compare to ``task.answer``. Symbolic equality via
    ``sympy`` when importable, otherwise a numeric/string fallback.
* ``mmlu`` / ``gpqa``
    Extract a single multiple-choice letter ``A-D`` (robust to phrasings such
    as ``"the answer is (B)"``, ``"B)"``, ``"B."``) and compare to
    ``task.answer``.
* ``livecodebench`` / ``bigcodebench``
    Execute candidate code against the task's tests in a subprocess sandbox
    with a timeout (``run_pass_at_1``). Never ``exec`` untrusted code in
    process.

Design contract
---------------
Every checker is a *pure* function of its inputs (no global state, no network)
so each can be unit-tested with one known-correct and one known-wrong case
(smoke test S5). The public entrypoint is :func:`score`.

This module has **no torch / GPU dependency** and imports only the stdlib plus
the shared :mod:`trinity.types`. ``sympy`` is imported lazily and guarded so the
module loads on a machine without it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from typing import Sequence

from trinity.types import Role, Task, Trajectory

__all__ = [
    "score",
    "score_text",
    "extract_boxed",
    "extract_last_number",
    "normalize_math_answer",
    "math_equal",
    "extract_choice_letter",
    "extract_code",
    "run_pass_at_1",
    "MATH_BENCHMARKS",
    "CHOICE_BENCHMARKS",
    "CODE_BENCHMARKS",
]

# Benchmark routing tables. Keys are matched case-insensitively against
# ``Task.benchmark`` (which the dataset loaders set, e.g. "math500").
MATH_BENCHMARKS: frozenset[str] = frozenset({"math500", "math", "aime", "aime2025"})
CHOICE_BENCHMARKS: frozenset[str] = frozenset({"mmlu", "gpqa", "gpqa-diamond", "gpqa_diamond"})
CODE_BENCHMARKS: frozenset[str] = frozenset(
    {"livecodebench", "lcb", "bigcodebench", "bigcode"}
)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def score(traj: Trajectory) -> float:
    """Return the binary reward ``R(tau) in {0.0, 1.0}`` for a trajectory.

    Dispatches on ``traj.task.benchmark``. The candidate answer is taken from
    ``traj.final_answer`` (the post-processed output of the terminating turn,
    ``O_tau``). For code benchmarks the candidate is the code extracted from the
    final answer and run against ``task.answer`` (the test spec).

    Args:
        traj: A completed :class:`~trinity.types.Trajectory` whose
            ``final_answer`` is populated and whose ``task`` carries the
            reference answer / test spec.

    Returns:
        ``1.0`` if the final answer is judged correct, else ``0.0``.

    Raises:
        ValueError: If the task's benchmark is not recognized.
    """
    benchmark = (traj.task.benchmark or "").strip().lower()
    ref = traj.task.answer
    candidate = _committed_answer(benchmark, traj)
    return score_text(benchmark, candidate, ref)


def _committed_answer(benchmark: str, traj: Trajectory) -> str:
    """Pick the text to score from a multi-turn trajectory.

    ``_final_answer`` (last Worker output) is often a verbose derivation with no
    cleanly-extractable answer, while an answer DID appear in some turn. To avoid
    throwing away answers the system actually produced, score the MOST RECENT turn
    whose output yields an extractable answer for this task type; fall back to
    ``final_answer``. This applies equally to TRINITY and the random baseline (the
    single-model baseline is one turn, so it is unaffected) — a fair fix, not a thumb
    on the scale. See JOURNAL 2026-06-23 (MMLU extraction diagnosis).
    """
    key = (benchmark or "").strip().lower()
    final = traj.final_answer or ""
    turns = getattr(traj, "turns", None) or []

    def has_answer(txt: str) -> bool:
        if not txt:
            return False
        if key in CHOICE_BENCHMARKS:
            return extract_choice_letter(txt) is not None
        if key in MATH_BENCHMARKS:
            return extract_boxed(txt) is not None or extract_last_number(txt) is not None
        if key in CODE_BENCHMARKS:
            return "```" in txt or "def " in txt or "import " in txt
        return False

    if has_answer(final):
        return final
    for tr in reversed(turns):
        txt = getattr(tr, "processed_output", "") or ""
        if has_answer(txt):
            return txt
    return final


def score_text(benchmark: str, candidate: str, reference: object) -> float:
    """Pure core of :func:`score`, decoupled from the Trajectory container.

    Useful for unit tests (S5): feed a benchmark name, a candidate string, and
    a reference answer directly.

    Args:
        benchmark: Benchmark identifier (case-insensitive), e.g. ``"math500"``.
        candidate: The model's final answer text (or code, for code tasks).
        reference: The reference answer. For math/choice this is the gold
            string; for code it is the test spec consumed by
            :func:`run_pass_at_1` (a list of tests, or a dict with ``tests`` and
            optional ``timeout_s``).

    Returns:
        ``1.0`` for correct, else ``0.0``.

    Raises:
        ValueError: If ``benchmark`` is not recognized.
    """
    key = (benchmark or "").strip().lower()
    if key in MATH_BENCHMARKS:
        return 1.0 if _check_math(candidate, reference) else 0.0
    if key in CHOICE_BENCHMARKS:
        return 1.0 if _check_choice(candidate, reference) else 0.0
    if key in CODE_BENCHMARKS:
        return 1.0 if _check_code(candidate, reference) else 0.0
    raise ValueError(
        f"Unknown benchmark {benchmark!r}. "
        f"Known: math={sorted(MATH_BENCHMARKS)}, "
        f"choice={sorted(CHOICE_BENCHMARKS)}, code={sorted(CODE_BENCHMARKS)}."
    )


# ---------------------------------------------------------------------------
# Math: MATH500 / AIME
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    r"""Extract the contents of the last ``\boxed{...}`` in ``text``.

    Handles nested braces by balanced-brace scanning (so ``\boxed{\frac{1}{2}}``
    returns ``\frac{1}{2}``). Returns the **last** boxed expression, since the
    final answer is conventionally boxed last.

    Args:
        text: Arbitrary model output that may contain LaTeX.

    Returns:
        The inner content of the last ``\boxed{...}`` (stripped), or ``None`` if
        no balanced ``\boxed{...}`` is present.
    """
    if not text:
        return None
    results: list[str] = []
    marker = r"\boxed"
    idx = 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        brace = pos + len(marker)
        # Skip whitespace between \boxed and the opening brace.
        while brace < len(text) and text[brace] in " \t":
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            idx = pos + len(marker)
            continue
        depth = 0
        start = brace + 1
        i = brace
        end = -1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end == -1:
            # Unbalanced; stop scanning further occurrences.
            break
        results.append(text[start:end].strip())
        idx = end + 1
    return results[-1] if results else None


def extract_last_number(text: str) -> str | None:
    """Extract the last numeric literal from ``text``.

    Used as a fallback when no ``\\boxed{...}`` answer is present. Recognizes
    integers, decimals, signed numbers, and thousands separators (commas are
    stripped). Trailing punctuation (a sentence-ending period) is not consumed
    as a decimal point.

    Args:
        text: Arbitrary model output.

    Returns:
        The last number as a string (commas removed), or ``None`` if no number
        is found.
    """
    if not text:
        return None
    # Match a simple fraction a/b FIRST (so "1/2" is kept whole, not read as "2"),
    # then decimals/integers like -1,234.56 or 42 or .5 ; require a digit somewhere.
    pattern = re.compile(
        r"-?\d+\s*/\s*-?\d+"
        r"|-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        r"|-?\.\d+"
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace(" ", "")


def normalize_math_answer(ans: str | None) -> str:
    r"""Normalize a math answer string for robust comparison.

    Strips LaTeX wrappers and cosmetic tokens that never change the value:
    ``$``/``\(``/``\)``, ``\left``/``\right``, ``\!``/``\,``/``\;``/``\:``,
    ``\text{...}``, ``\%`` and trailing ``%``, ``^\circ``/``\degree``, a leading
    ``=``, surrounding ``\{...\}``, and outer whitespace. Collapses internal
    whitespace and lowercases. Converts ``a/b`` integer fractions and
    ``\frac{a}{b}`` to a canonical ``Fraction`` string when possible.

    Args:
        ans: Raw answer text (or ``None``).

    Returns:
        A normalized string suitable for exact comparison (empty string for
        ``None``).
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    # Drop a leading "answer:" style prefix.
    s = re.sub(r"^(the\s+)?(final\s+)?answer(\s+is)?\s*[:=]?\s*", "", s, flags=re.I)
    # Remove math-mode delimiters.
    for tok in ("$", r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\(", r"\)"):
        s = s.replace(tok, "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"^\circ", "").replace(r"\degree", "")
    s = s.replace(r"\$", "")
    s = s.strip()
    if s.startswith("="):
        s = s[1:].strip()
    # Strip a single outer pair of \{ \} or { }.
    s = re.sub(r"^\\?\{(.*)\\?\}$", r"\1", s).strip()
    # \frac{a}{b} -> a/b
    s = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\d?frac\s*(\d)\s*(\d)", r"\1/\2", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    # Canonicalize a pure integer ratio a/b.
    m = re.fullmatch(r"\(?(-?\d+)\)?/\(?(-?\d+)\)?", s)
    if m:
        try:
            return str(Fraction(int(m.group(1)), int(m.group(2))))
        except (ZeroDivisionError, ValueError):
            pass
    return s


def _as_number(s: str) -> float | None:
    """Best-effort parse of a normalized string to a float, else ``None``."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m:
        try:
            denom = float(m.group(2))
            if denom != 0.0:
                return float(m.group(1)) / denom
        except ValueError:
            return None
    return None


def math_equal(a: str | None, b: str | None, *, rel_tol: float = 1e-6) -> bool:
    """Compare two math answers for equality.

    Resolution order:
      1. Exact match after :func:`normalize_math_answer`.
      2. Numeric match within ``rel_tol`` (handles ``0.5`` vs ``1/2`` etc.).
      3. Symbolic equality via ``sympy`` if it is importable (guarded).

    Args:
        a: First answer (typically the candidate).
        b: Second answer (typically the reference).
        rel_tol: Relative tolerance for the numeric comparison.

    Returns:
        ``True`` if the two answers are judged equal.
    """
    na = normalize_math_answer(a)
    nb = normalize_math_answer(b)
    if na == nb and na != "":
        return True

    fa = _as_number(na)
    fb = _as_number(nb)
    if fa is not None and fb is not None:
        scale = max(1.0, abs(fa), abs(fb))
        if abs(fa - fb) <= rel_tol * scale:
            return True

    return _sympy_equal(na, nb)


def _sympy_equal(a: str, b: str) -> bool:
    """Symbolic-equality fallback. Returns ``False`` if sympy is unavailable."""
    if not a or not b:
        return False
    try:  # guarded import: local machine may lack sympy
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except Exception:
        return False
    transformations = standard_transformations + (
        implicit_multiplication_application,
    )
    try:
        ea = parse_expr(a, transformations=transformations, evaluate=True)
        eb = parse_expr(b, transformations=transformations, evaluate=True)
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def _check_math(candidate: str, reference: object) -> bool:
    """True iff the candidate's extracted answer equals the reference."""
    extracted = extract_boxed(candidate)
    if extracted is None:
        extracted = extract_last_number(candidate)
    if extracted is None:
        # Last resort: compare the whole (normalized) candidate.
        extracted = candidate

    ref_str = reference if isinstance(reference, str) else _ref_to_str(reference)
    # The reference itself may be boxed (datasets vary).
    ref_boxed = extract_boxed(ref_str)
    if ref_boxed is not None:
        ref_str = ref_boxed
    return math_equal(extracted, ref_str)


def _ref_to_str(reference: object) -> str:
    """Coerce a non-string reference (int/float/Fraction) to text."""
    if reference is None:
        return ""
    return str(reference)


# ---------------------------------------------------------------------------
# Multiple choice: MMLU / GPQA
# ---------------------------------------------------------------------------
# Match in priority order. Earlier patterns are more explicit / trustworthy.
_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Require the captured letter to be followed by a delimiter or end-of-word,
    # so "the answer Beats..." does NOT match "B" (P2 review fix).
    re.compile(r"answer\s*(?:is|:)?\s*\(?\s*([A-D])\s*(?:[\).:]|\b)(?![A-Za-z])", re.I),
    re.compile(r"\\boxed\s*\{\s*\(?\s*([A-D])\s*\)?\s*\}", re.I),
    re.compile(r"\bfinal\s+answer\s*[:=]?\s*\(?\s*([A-D])(?![A-Za-z])", re.I),
    re.compile(r"\boption\s*\(?\s*([A-D])(?![A-Za-z])", re.I),
    re.compile(r"^\s*\(?\s*([A-D])\s*[\).:]", re.M),
)


def extract_choice_letter(text: str) -> str | None:
    """Extract a single multiple-choice letter ``A``-``D`` from ``text``.

    Robust to common phrasings: ``"the answer is (B)"``, ``"Answer: C"``,
    ``"B)"``, ``"B."``, ``"\\boxed{D}"``, ``"Option A"``. Tries explicit
    answer-bearing patterns first; if none match, falls back to the **last**
    standalone capital ``A``-``D`` token in the text (final answers usually come
    last). Letters embedded in words (e.g. the ``A`` in ``"And"``) are excluded
    by requiring word boundaries / delimiters.

    Args:
        text: Arbitrary model output.

    Returns:
        The uppercase letter, or ``None`` if no choice can be identified.
    """
    if not text:
        return None
    for pat in _CHOICE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    # Fallback (P2 review fix): only trust the LAST non-empty line, and only when
    # it is essentially just the letter (e.g. "B", "(C)", "D."). This avoids the
    # English article "A" in prose like "A nice approach" being read as a choice.
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        m = re.fullmatch(r"\(?\s*([A-D])\s*\)?[.:]?", line, re.I)
        if m:
            return m.group(1).upper()
        break  # only inspect the final non-empty line
    return None


def _check_choice(candidate: str, reference: object) -> bool:
    """True iff the extracted letter matches the reference letter."""
    got = extract_choice_letter(candidate)
    if got is None:
        return False
    ref = _normalize_reference_letter(reference)
    if ref is None:
        return False
    return got == ref


def _normalize_reference_letter(reference: object) -> str | None:
    """Coerce a reference answer to a single ``A``-``D`` letter.

    Accepts a letter string (``"B"``, ``"(B)"``) or a 0-based / 1-based integer
    index (``1`` -> ``"B"`` under 0-based; datasets vary, so a bare letter is
    preferred). Returns ``None`` if it cannot be resolved.
    """
    if reference is None:
        return None
    if isinstance(reference, str):
        letter = extract_choice_letter(reference)
        if letter is not None:
            return letter
        s = reference.strip().upper()
        return s if s in {"A", "B", "C", "D"} else None
    if isinstance(reference, bool):
        return None
    if isinstance(reference, int):
        if 0 <= reference <= 3:
            return "ABCD"[reference]
        return None
    return None


# ---------------------------------------------------------------------------
# Code: LiveCodeBench / BigCodeBench
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(
    r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_code(text: str) -> str:
    """Extract a Python code block from model output.

    If the text contains one or more fenced code blocks (```` ```python ... ```
    ````), the **last** such block is returned (the final solution usually comes
    last). If no fence is present, the text is returned verbatim (stripped),
    assuming the whole output is code.

    Args:
        text: Model output that may wrap code in Markdown fences.

    Returns:
        The extracted source code (without the fence markers).
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    return text.strip()


def _coerce_test_spec(reference: object) -> tuple[list, int]:
    """Normalize a code reference into ``(tests, timeout_s)``.

    The ``Task.answer`` for code benchmarks may be:
      * a ``list`` of tests, or
      * a ``dict`` with key ``"tests"`` and optional ``"timeout_s"``, or
      * a JSON string encoding either of the above.

    Each test is one of:
      * a ``str`` of assert-based Python (executed after the candidate code), or
      * a ``dict`` ``{"stdin": str, "expected_stdout": str}`` for I/O tests, or
      * a 2-tuple/list ``(stdin, expected_stdout)``.
    """
    timeout_s = 10
    spec: object = reference
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (json.JSONDecodeError, ValueError):
            spec = [spec]
    if isinstance(spec, dict):
        timeout_s = int(spec.get("timeout_s", timeout_s))
        tests = spec.get("tests", [])
    else:
        tests = spec
    if tests is None:
        tests = []
    if not isinstance(tests, list):
        tests = [tests]
    return tests, timeout_s


def _check_code(candidate: str, reference: object) -> bool:
    """True iff extracted code passes all tests in the reference spec."""
    code = extract_code(candidate)
    if not code.strip():
        return False
    tests, timeout_s = _coerce_test_spec(reference)
    return run_pass_at_1(code, tests, timeout_s=timeout_s)


def run_pass_at_1(code: str, tests: Sequence, timeout_s: int = 10) -> bool:
    """Execute candidate ``code`` against ``tests`` in a subprocess sandbox.

    The candidate code is **never** executed in-process. Each invocation writes
    a temporary script and runs it with the current Python interpreter in a
    fresh subprocess with a wall-clock timeout. The candidate is judged to pass
    only if **every** test passes.

    Two test flavors are supported (they may be mixed in one list):

    * **assert-based** (``str``): arbitrary Python appended after the candidate
      code; a test passes if the script exits ``0`` with no exception. Use this
      for function-call style benchmarks (BigCodeBench).
    * **stdin/stdout** (``dict`` with ``"stdin"`` / ``"expected_stdout"`` or a
      ``(stdin, expected_stdout)`` pair): the candidate is run as a program, fed
      ``stdin`` on standard input, and its stdout is compared (whitespace-
      trimmed per line) to ``expected_stdout``. Use this for competitive-
      programming style benchmarks (LiveCodeBench).

    Args:
        code: Candidate Python source (already fence-stripped).
        tests: Sequence of tests as described above.
        timeout_s: Per-test wall-clock timeout in seconds.

    Returns:
        ``True`` iff the candidate passes all tests (and there is at least one
        test). An empty test list returns ``False`` (nothing was verified).
    """
    if not code.strip():
        return False
    if not tests:
        return False
    for test in tests:
        if not _run_one_test(code, test, timeout_s):
            return False
    return True


def _run_one_test(code: str, test: object, timeout_s: int) -> bool:
    """Run a single test in an isolated subprocess. Returns pass/fail."""
    stdin_data: str | None = None
    expected_stdout: str | None = None
    assert_block: str | None = None

    if isinstance(test, dict):
        if (
            "stdin" in test
            or "input" in test
            or "expected_stdout" in test
            or "output" in test
        ):
            # dataset.py emits LiveCodeBench tests as {"input": ..., "output": ...};
            # accept both key conventions so stdin is never silently empty.
            stdin_data = str(test.get("stdin", test.get("input", "")))
            expected_stdout = str(
                test.get("expected_stdout", test.get("output", ""))
            )
        elif "assert" in test:
            assert_block = str(test["assert"])
        else:
            # Unknown dict shape — treat any "test"/"code" field as assert code.
            assert_block = str(test.get("test", test.get("code", "")))
    elif isinstance(test, (tuple, list)) and len(test) == 2:
        stdin_data = str(test[0])
        expected_stdout = str(test[1])
    elif isinstance(test, str):
        assert_block = test
    else:
        return False

    if assert_block is not None:
        script = code + "\n\n" + assert_block + "\n"
        return _exec_script(script, stdin_data="", timeout_s=timeout_s)

    # stdin/stdout test.
    ok, stdout = _exec_script_capture(
        code, stdin_data=stdin_data or "", timeout_s=timeout_s
    )
    if not ok:
        return False
    return _stdout_matches(stdout, expected_stdout or "")


def _stdout_matches(got: str, expected: str) -> bool:
    """Compare program output to expected, ignoring trailing whitespace."""
    got_lines = [ln.rstrip() for ln in got.replace("\r\n", "\n").rstrip().split("\n")]
    exp_lines = [
        ln.rstrip() for ln in expected.replace("\r\n", "\n").rstrip().split("\n")
    ]
    return got_lines == exp_lines


def _sandbox_env() -> dict[str, str]:
    """Minimal environment for the child interpreter."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    # Preserve a HOME so libraries that need a writable dir do not crash.
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def _exec_script(script: str, *, stdin_data: str, timeout_s: int) -> bool:
    """Run a script; pass iff it exits 0 within the timeout. No output check."""
    ok, _ = _exec_script_capture(script, stdin_data=stdin_data, timeout_s=timeout_s)
    return ok


def _exec_script_capture(
    script: str, *, stdin_data: str, timeout_s: int
) -> tuple[bool, str]:
    """Run a script in a subprocess and capture stdout.

    Args:
        script: The full Python source to execute.
        stdin_data: Data piped to the child's standard input.
        timeout_s: Wall-clock timeout in seconds.

    Returns:
        ``(ok, stdout)`` where ``ok`` is ``True`` iff the process exited with
        return code ``0`` (no exception/timeout) and ``stdout`` is the captured
        standard output (empty on failure).
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(script)
            tmp_path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=_sandbox_env(),
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            return False, ""
        except (OSError, ValueError):
            return False, ""
        return (proc.returncode == 0), (proc.stdout or "")
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Convenience: keep Role import meaningful for downstream type checks.
# ---------------------------------------------------------------------------
def _terminating_role(traj: Trajectory) -> Role | None:
    """Return the role of the terminating turn, or ``None`` if no turns.

    Exposed for orchestration/debugging: a Verifier-ACCEPT terminated run ends
    on a :class:`~trinity.types.Role.VERIFIER` turn, but the scored answer is
    the last non-verifier ``O_k`` carried in ``final_answer``.
    """
    if not traj.turns:
        return None
    return traj.turns[-1].role
