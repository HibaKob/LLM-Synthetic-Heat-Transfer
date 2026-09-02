"""
Tests for verification.py

Includes positive and negative cases
"""

import pytest
from syntheticheatransfer import verification as vf 


###############################################################################
# Type A
###############################################################################

def test_type_a_correct_passes():
    problem = {
        "id": "A-ok", "type": "A",
        "answer_numeric": 2000.0, "answer_tolerance_rel": 0.01,
        "check_expression": "1.2*(300-50)/0.15",
    }
    assert vf.verify_type_a(problem).passed


def test_type_a_wrong_answer_fails():
    # answer_numeric disagrees with the expression -> must fail.
    problem = {
        "id": "A-bad", "type": "A",
        "answer_numeric": 9999.0, "answer_tolerance_rel": 0.01,
        "check_expression": "1.2*(300-50)/0.15",
    }
    assert not vf.verify_type_a(problem).passed


def test_type_a_uses_math_module():
    problem = {
        "id": "A-math", "type": "A",
        "answer_numeric": 13945.0, "answer_tolerance_rel": 0.02,
        "check_expression": "2*math.pi*15*1*(100-40)/math.log(0.03/0.02)",
    }
    assert vf.verify_type_a(problem).passed


def test_type_a_malformed_expression_fails_gracefully():
    problem = {
        "id": "A-err", "type": "A",
        "answer_numeric": 1.0, "answer_tolerance_rel": 0.01,
        "check_expression": "1/0",
    }
    res = vf.verify_type_a(problem)
    assert not res.passed
    assert "failed" in res.detail.lower()


def test_type_a_missing_field_fails():
    problem = {"id": "A-missing", "type": "A", "answer_numeric": 1.0}
    assert not vf.verify_type_a(problem).passed


###############################################################################
# Type B
###############################################################################

REF = (
    "import numpy as np\n"
    "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
    "    A = np.zeros((n, n)); b = np.zeros(n)\n"
    "    A[0,0]=1.0; b[0]=T_left; A[-1,-1]=1.0; b[-1]=T_right\n"
    "    for i in range(1, n-1):\n"
    "        A[i,i-1]=1.0; A[i,i]=-2.0; A[i,i+1]=1.0; b[i]=0.0\n"
    "    return list(np.linalg.solve(A, b))\n"
)

PROBLEM_B = {
    "id": "B-cond", "type": "B",
    "function_name": "solve_1d_conduction",
    "reference_solution": REF,
    "test_cases": [
        {"name": "linear_100_0", "args": [1, 11, 100, 0, 1],
         "expected": [100 - 10*i for i in range(11)], "tol_rel": 1e-6},
        {"name": "linear_50_150", "args": [2, 5, 50, 150, 10],
         "expected": [50 + 25*i for i in range(5)], "tol_rel": 1e-6},
    ],
}


def test_type_b_reference_self_consistent():
    # Reference verified against itself.
    assert vf.verify_type_b(PROBLEM_B).passed


def test_type_b_correct_candidate_passes():
    # A different-but-correct implementation should pass.
    candidate = (
        "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
        "    return [T_left + (T_right - T_left)*i/(n-1) for i in range(n)]\n"
    )
    assert vf.verify_type_b(PROBLEM_B, candidate_solution=candidate).passed


def test_type_b_wrong_candidate_fails():
    # Returns zeros -> must fail.
    candidate = (
        "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
        "    return [0.0]*n\n"
    )
    assert not vf.verify_type_b(PROBLEM_B, candidate_solution=candidate).passed


def test_type_b_candidate_wrong_function_name_fails():
    candidate = "def wrong_name(L, n, T_left, T_right, k):\n    return []\n"
    assert not vf.verify_type_b(PROBLEM_B, candidate_solution=candidate).passed


def test_type_b_candidate_runtime_error_fails():
    candidate = (
        "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
        "    raise ValueError('boom')\n"
    )
    res = vf.verify_type_b(PROBLEM_B, candidate_solution=candidate)
    assert not res.passed


def test_type_b_disallowed_import_fails():
    # Attempting to import os must be blocked -> load fails -> not passed.
    candidate = (
        "import os\n"
        "def solve_1d_conduction(L, n, T_left, T_right, k):\n"
        "    return [0.0]*n\n"
    )
    assert not vf.verify_type_b(PROBLEM_B, candidate_solution=candidate).passed


def test_type_b_missing_field_fails():
    problem = {"id": "B-missing", "type": "B", "function_name": "f"}
    # no reference_solution / test_cases
    assert not vf.verify_type_b(problem).passed


def test_values_close_type_mismatch_fails():
    # returns a scalar where the expected value is a list
    ref = "def f(n):\n    return [1.0]*n\n"
    problem = {"id": "mismatch", "type": "B", "function_name": "f",
               "reference_solution": ref,
               "test_cases": [{"name": "c", "args": [3], "expected": 1.0, "tol_rel": 1e-6}]}
    assert not vf.verify_type_b(problem).passed


###############################################################################
# Public helpers
###############################################################################

def test_expression_namespace_has_math():
    ns = vf.expression_namespace()
    assert "math" in ns
    assert ns["__builtins__"] == {}     


def test_expression_namespace_numpy_if_available():
    ns = vf.expression_namespace()
    try:
        import numpy
        assert "np" in ns and "numpy" in ns
    except ImportError:
        assert "np" not in ns


def test_expression_namespace_evaluates_math_expr():
    ns = vf.expression_namespace()
    assert abs(eval("math.sqrt(200)", ns) - 14.142135) < 1e-5


def test_load_solution_returns_callable():
    fn = vf.load_solution("def f(x):\n    return x * 2\n", "f")
    assert callable(fn)
    assert fn(21) == 42


def test_load_solution_missing_function_raises():
    with pytest.raises(Exception):
        vf.load_solution("def other():\n    return 1\n", "not_defined")


def test_create_output_folder(tmp_path):
    folder = vf.create_output_folder(tmp_path, "output")
    assert folder.is_dir()

def test_create_output_folder_preexist(tmp_path):
    vf.create_output_folder(tmp_path, "output")
    folder = vf.create_output_folder(tmp_path, "output")
    assert folder.is_dir()

def test_create_output_folder_creates_parents(tmp_path):
    folder = vf.create_output_folder(tmp_path / "missing" / "parents", "leaf")
    assert folder.is_dir()


###############################################################################
# Run
###############################################################################

def test_dispatch_unknown_type_fails():
    assert not vf.verify_problem({"id": "x", "type": "Z"}).passed


def test_verify_batch_counts():
    good = {"id": "g", "type": "A", "answer_numeric": 4.0,
            "answer_tolerance_rel": 0.01, "check_expression": "2*2"}
    bad = {"id": "b", "type": "A", "answer_numeric": 5.0,
           "answer_tolerance_rel": 0.01, "check_expression": "2*2"}
    summary = vf.verify_batch([good, bad, good])
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
