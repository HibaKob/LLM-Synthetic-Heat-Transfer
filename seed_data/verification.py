"""
Deterministic verification for synthetic engineering problems.

Two problem types are supported, matching the seed schema:

  Type A (analytical): the problem carries a `check_expression` (a pure Python
      expression using only the `math` module) that recomputes the answer
      independently. Verification recomputes it and compares to `answer_numeric`
      within `answer_tolerance_rel`.

  Type B (coding): the problem carries a `reference_solution` (source code that
      defines a function) and a list of `test_cases`. Verification executes the
      candidate solution in a restricted namespace and checks each test case,
      comparing against the reference solution's output (or an explicit expected
      value) within a tolerance.

The point of this module is that correctness is established by COMPUTATION, not by
trusting an LLM. It is the deterministic quality filter in the generation pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class VerificationResult:
    """Outcome of verifying a single problem."""
    passed: bool
    problem_id: Optional[str] = None
    problem_type: Optional[str] = None
    detail: str = ""
    # per-test breakdown for Type B (list of (name, passed, message))
    test_results: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rel_close(value: float, expected: float, tol_rel: float) -> bool:
    """Relative-tolerance comparison, robust near zero."""
    denom = max(abs(expected), 1e-12)
    return abs(value - expected) <= tol_rel * denom


def _expression_namespace() -> dict:
    """The namespace available to Type A check_expressions. Single source of truth."""
    ns = {"__builtins__": {}, "math": math}
    try:
        import numpy as np
        ns["np"] = np
        ns["numpy"] = np
    except ImportError:
        pass
    return ns


def _eval_expression(expr: str) -> float:
    """
    Evaluate a Type A check_expression in a namespace defined by '_expression_namespace'.
    """
    return eval(expr, _expression_namespace())


def _safe_exec(source: str) -> dict:
    """
    Execute `source` (a function definition) in a namespace with a restricted set
    of builtins plus math/numpy, and return the resulting namespace.
    """
    # Whitelist of modules a solution may import.
    _allowed_modules = {"math": math}
    try:
        import numpy as _np
        _allowed_modules["numpy"] = _np
    except Exception:
        _np = None

    def _guarded_import(name, *args, **kwargs):
        if name in _allowed_modules:
            return _allowed_modules[name]
        raise ImportError(f"import of '{name}' is not allowed in the verification sandbox")

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "range": range,
        "len": len, "enumerate": enumerate, "zip": zip, "map": map,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "float": float, "int": int, "bool": bool, "round": round,
        "sorted": sorted, "reversed": reversed, "print": print,
        "__import__": _guarded_import,
    }
    ns: dict = {"__builtins__": safe_builtins, "math": math}
    if _np is not None:
        ns["np"] = _np
        ns["numpy"] = _np
    exec(source, ns)  # noqa: S102 - source is generated/controlled, run in a sandbox
    return ns


def _get_callable(ns: dict, func_name: str) -> Callable:
    fn = ns.get(func_name)
    if not callable(fn):
        raise ValueError(f"Function '{func_name}' not defined by the solution.")
    return fn


def _values_close(got: Any, expected: Any, tol_rel: float) -> bool:
    """
    Compare scalars or 1-D sequences element-wise with relative tolerance.
    Extend as needed for nested structures.
    """
    # sequence (list/tuple) case
    if isinstance(expected, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(expected):
            return False
        return all(_rel_close(float(g), float(e), tol_rel) for g, e in zip(got, expected))
    # scalar case
    return _rel_close(float(got), float(expected), tol_rel)


# --------------------------------------------------------------------------- #
# Type A: analytical / numeric verification
# --------------------------------------------------------------------------- #

def verify_type_a(problem: dict) -> VerificationResult:
    """
    Verify an analytical problem.

    Expects keys: id, type=='A', answer_numeric, answer_tolerance_rel, check_expression.
    Recomputes check_expression and compares to answer_numeric.

    This confirms the stated answer is internally consistent with an independent
    computation. At this point, it does not, by itself, confirm the problem is well-posed.
    (can be paired with an LLM-judge for reasoning quality if desired)
    """
    pid = problem.get("id")
    try:
        expr = problem["check_expression"]
        expected = float(problem["answer_numeric"])
        tol = float(problem.get("answer_tolerance_rel", 0.01))
    except KeyError as e:
        return VerificationResult(False, pid, "A", f"Missing field: {e}")

    try:
        computed = float(_eval_expression(expr))
    except Exception as e:
        return VerificationResult(False, pid, "A", f"check_expression failed: {e!r}")

    if _rel_close(computed, expected, tol):
        return VerificationResult(
            True, pid, "A",
            f"OK: computed {computed:.6g} matches answer {expected:.6g} (rel tol {tol}).",
        )
    return VerificationResult(
        False, pid, "A",
        f"MISMATCH: computed {computed:.6g} vs stated {expected:.6g} (rel tol {tol}).",
    )


# --------------------------------------------------------------------------- #
# Type B: code-execution verification
# --------------------------------------------------------------------------- #

def verify_type_b(
    problem: dict,
    candidate_solution: Optional[str] = None,
) -> VerificationResult:
    """
    Verify a coding problem by executing code against test cases.

    Expects keys: id, type=='B', function_name, reference_solution, test_cases.
      test_cases: list of dicts, each with:
        - "args": list/tuple of positional arguments
        - optionally "expected": explicit expected return value
        - optionally "tol_rel": per-case tolerance (default 1e-6)
      If a test case has no "expected", the reference_solution's output is used as
      ground truth (i.e. the candidate must match the reference).

    `candidate_solution`: source code to verify. If None, the reference_solution is
    verified against itself (useful for validating that seeds are self-consistent).

    """
    pid = problem.get("id")
    try:
        func_name = problem["function_name"]
        reference = problem["reference_solution"]
        test_cases = problem["test_cases"]
    except KeyError as e:
        return VerificationResult(False, pid, "B", f"Missing field: {e}")

    source = candidate_solution if candidate_solution is not None else reference

    # Load candidate function
    try:
        cand_ns = _safe_exec(source)
        cand_fn = _get_callable(cand_ns, func_name)
    except Exception as e:
        return VerificationResult(False, pid, "B", f"Candidate failed to load: {e!r}")

    # Load reference function (for cases without explicit expected values)
    try:
        ref_ns = _safe_exec(reference)
        ref_fn = _get_callable(ref_ns, func_name)
    except Exception as e:
        return VerificationResult(False, pid, "B", f"Reference failed to load: {e!r}")

    test_results = []
    all_passed = True

    for i, case in enumerate(test_cases):
        name = case.get("name", f"case_{i+1}")
        args = list(case.get("args", []))
        tol = float(case.get("tol_rel", 1e-6))
        try:
            expected = case["expected"] if "expected" in case else ref_fn(*args)
            got = cand_fn(*args)
            ok = _values_close(got, expected, tol)
            msg = "ok" if ok else f"got {got!r} vs expected {expected!r}"
        except Exception as e:
            ok = False
            msg = f"execution error: {e!r}"
        all_passed = all_passed and ok
        test_results.append((name, ok, msg))

    detail = "; ".join(f"{n}: {m}" for n, ok, m in test_results)
    return VerificationResult(all_passed, pid, "B", detail, test_results)


# --------------------------------------------------------------------------- #
# Dispatch + batch
# --------------------------------------------------------------------------- #

def verify_problem(
    problem: dict,
    candidate_solution: Optional[str] = None,
) -> VerificationResult:
    """Dispatch to the correct verifier based on problem['type'] ('A' or 'B')."""
    ptype = problem.get("type")
    if ptype == "A":
        return verify_type_a(problem)
    if ptype == "B":
        return verify_type_b(problem, candidate_solution)
    return VerificationResult(False, problem.get("id"), ptype, f"Unknown type: {ptype!r}")


def verify_batch(problems: list) -> dict:
    """
    Verify a list of problems. Returns a summary dict:
      {"total", "passed", "failed", "results": [VerificationResult, ...]}
    Useful for filtering a generated batch: keep [p for p, r in zip(problems, results) if r.passed].
    """
    results = [verify_problem(p) for p in problems]
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


if __name__ == "__main__":
    # Testing with inline examples matching the seed schema.
    a = {
        "id": "demo-A", "type": "A",
        "answer_numeric": 2000.0, "answer_tolerance_rel": 0.01,
        "check_expression": "1.2*(300-50)/0.15",
    }
    b = {
        "id": "demo-B", "type": "B",
        "function_name": "solve_1d_conduction",
        "reference_solution": (
            "import numpy as np\n"
            "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
            "    A = np.zeros((n, n)); b = np.zeros(n)\n"
            "    A[0,0]=1.0; b[0]=T_left; A[-1,-1]=1.0; b[-1]=T_right\n"
            "    for i in range(1, n-1):\n"
            "        A[i,i-1]=1.0; A[i,i]=-2.0; A[i,i+1]=1.0; b[i]=0.0\n"
            "    return list(np.linalg.solve(A, b))\n"
        ),
        "test_cases": [
            {"name": "linear_100_0", "args": [1, 11, 100, 0, 1],
             "expected": [100 - 10*i for i in range(11)], "tol_rel": 1e-6},
            {"name": "linear_50_150", "args": [2, 5, 50, 150, 10],
             "expected": [50 + 25*i for i in range(5)], "tol_rel": 1e-6},
        ],
    }
    for prob in (a, b):
        r = verify_problem(prob)
        print(f"[{r.problem_id}] {'PASS' if r.passed else 'FAIL'} - {r.detail}")
