"""
Tests for quality.py (semantic quality checks).

Three kinds of tests:
  * Unit tests: each deterministic check on small hand-made problems, one that
    should pass and one that should fail.
  * Regression tests against the Nov 2026 audit: the 80 generated problems in
    tests/fixtures/generated_examples_2026-11-02.json were labelled by hand
    (tests/fixtures/audit_labels.json). The checks must reject none of the
    correct ones and keep catching at least as many wrong ones as they do now.
  * Re-solve tests: the second-model call is replaced by a fake `chat` function,
    so these run offline.
"""

import json
from pathlib import Path

import pytest

from syntheticheatransfer import quality as q

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = REPO_ROOT / "data" / "seeds" / "heat_transfer_seeds.json"


def type_a(instruction, expr, units, answer=None, pid="t-A"):
    from syntheticheatransfer import verification as v
    if answer is None:
        answer = float(eval(expr, v.expression_namespace()))
    return {"id": pid, "type": "A", "instruction": instruction, "check_expression": expr,
            "answer_units": units, "answer_numeric": answer, "answer_tolerance_rel": 0.02}


def type_b(instruction, code, fn, cases, pid="t-B"):
    from syntheticheatransfer import verification as v
    f = v.load_solution(code, fn)
    return {"id": pid, "type": "B", "instruction": instruction, "function_name": fn,
            "reference_solution": code,
            "test_cases": [{"name": f"case_{i}", "args": a, "expected": f(*a), "tol_rel": 1e-6}
                           for i, a in enumerate(cases)]}


WALL = ("A plane wall of thickness 0.2 m and area 3 m^2 has thermal conductivity "
        "0.6 W/m*K. Its surfaces are at 25 C and 15 C. Find the heat transfer rate in W.")


###############################################################################
# Type A checks
###############################################################################

def test_correct_problem_passes_every_check():
    report = q.check_problem(type_a(WALL, "0.6*3*(25-15)/0.2", "W"))
    assert report.passed, report.summary()


def test_asks_question():
    assert q.check_a_asks_question(type_a(WALL, "1", "W")).passed
    p = type_a("A wall has k = 0.6 W/m*K and is 0.2 m thick.", "0.6/0.2", "W/m^2*K")
    assert not q.check_a_asks_question(p).passed


def test_negative_rate_fails():
    p = type_a(WALL, "0.6*3*(15-25)/0.2", "W")
    assert not q.check_a_positive_magnitude(p).passed


def test_negative_temperature_allowed():
    p = type_a("Find the temperature in C after cooling.", "-5.0", "C")
    assert q.check_a_positive_magnitude(p).skipped


def test_zero_answer_fails():
    assert not q.check_a_finite_nonzero(type_a(WALL, "0*1", "W")).passed


def test_invented_number_fails_grounding():
    # k = 386 (copper) is not given in the text.
    text = "A copper rod 1 m long and 0.02 m in diameter has ends at 80 C and 50 C. Find the heat rate in W."
    r = q.check_a_numbers_grounded(type_a(text, "386*math.pi*0.02**2/4*(80-50)/1", "W"))
    assert not r.passed and "386" in r.detail


def test_grounding_allows_conversions_and_constants():
    text = ("A pipe of diameter 30 mm loses heat with h = 6 W/m^2*K. Its surface is at 80 C "
            "and the air at 15 C. Find the heat loss per meter length in W/m.")
    assert q.check_a_numbers_grounded(type_a(text, "6*(2*math.pi*(30/2)*1e-3)*(80-15)", "W/m")).passed
    text_k = "A black surface at 227 C radiates. Find its emissive power in W/m^2."
    assert q.check_a_numbers_grounded(type_a(text_k, "5.67e-8*(227+273.15)**4", "W/m^2")).passed


def test_grounding_reads_scientific_notation():
    text = "Air (nu = 1.5*10^-5 m^2/s) flows at 10 m/s over a 2 m plate. Find the Reynolds number."
    assert q.check_a_numbers_grounded(type_a(text, "10*2/(1.5*1e-5)", "dimensionless")).passed


def test_precedence_bug_fails():
    text = ("A sphere of diameter 0.15 m at 250 C is in water at 20 C with h = 100 W/m^2*K. "
            "Find the heat transfer rate in W.")
    assert not q.check_a_precedence(type_a(text, "100*(math.pi*0.15**2)*250-20", "W")).passed
    assert not q.check_a_precedence(type_a(text, "250-20/(1/(100*math.pi*0.15**2))", "W")).passed
    assert q.check_a_precedence(type_a(text, "100*(math.pi*0.15**2)*(250-20)", "W")).passed


def test_precedence_ignores_sums_of_terms_and_powers():
    assert q.check_a_precedence(type_a(WALL, "1/12 + 0.004/0.25 + 1/8.0", "W")).passed
    assert q.check_a_precedence(type_a(WALL, "5.67e-8*(400**4 - 300**4)", "W")).passed


def test_units_match_question():
    p = type_a("A plate loses 1200 W from 2 m^2 at 60 C to 20 C air. Find the convection "
               "coefficient h.", "1200/(2*(60-20))", "W")
    assert not q.check_a_units_match_question(p).passed


def test_dimensions_catch_missing_thickness():
    # k*A*dT has units W*m, not W: the thickness is missing.
    text = ("A heat exchanger wall has area 10 m^2, thermal conductivity 400 W/m*K and a "
            "temperature difference of 20 K. Find the heat transfer rate in W.")
    r = q.check_a_dimensions(type_a(text, "400*10*20", "W"))
    assert not r.passed


def test_dimensions_catch_adding_watts_to_kelvin():
    text = ("A sphere of diameter 0.15 m at 250 C is in water at 20 C with h = 100 W/m^2*K. "
            "Find the heat transfer rate in W.")
    r = q.check_a_dimensions(type_a(text, "100*(math.pi*0.15**2)*250-20", "W"))
    assert not r.passed


def test_dimensions_pass_on_correct_formula():
    assert q.check_a_dimensions(type_a(WALL, "0.6*3*(25-15)/0.2", "W")).passed


def test_dimensions_skip_when_a_unit_is_unknown():
    text = "A rod carries a current of 10 A. Find the heat rate in W."
    assert q.check_a_dimensions(type_a(text, "10**2*0.5", "W")).skipped


def test_parse_units():
    w = q.parse_units("W")
    assert q.parse_units("W/m^2*K") == q.parse_units("W/m²K") == q.parse_units("W/m2K")
    assert q.parse_units("J/kg*K") == q._dim(0, 2, -2, -1)
    assert q.parse_units("dimensionless") == q.DIMENSIONLESS
    assert q.parse_units("furlongs") is None
    assert w == q._dim(1, 2, -3, 0)


def test_plausible_range():
    text = "Find the convective heat transfer coefficient h in W/m^2*K."
    assert not q.check_a_plausible_range(type_a(text, "2.7e8", "W/m^2*K")).passed
    assert q.check_a_plausible_range(type_a(text, "25.0", "W/m^2*K")).passed
    vf = "Find the view factor F12."
    assert not q.check_a_plausible_range(type_a(vf, "1.4", "dimensionless")).passed


def test_circular_area():
    text = ("A copper rod of diameter 0.02 m and length 1 m (k = 386 W/m*K) has ends at 80 C "
            "and 50 C. Find the heat transfer rate in W.")
    assert not q.check_a_circular_area(type_a(text, "386*math.pi*0.02**2*(80-50)/1", "W")).passed
    assert q.check_a_circular_area(type_a(text, "386*math.pi*0.02**2/4*(80-50)/1", "W")).passed
    sphere = "A sphere of diameter 0.15 m ... Find the heat transfer rate in W."
    assert q.check_a_circular_area(type_a(sphere, "100*math.pi*0.15**2*230", "W")).skipped


def test_a_crashing_check_is_reported_not_raised():
    p = type_a(WALL, "1", "W")
    p["check_expression"] = "((("  # cannot be parsed
    report = q.check_problem(p)
    assert not report.passed


###############################################################################
# Type B checks
###############################################################################

PROFILE_TEXT = ("Write a Python function f(L, n, T_left, T_right) that solves steady 1-D conduction. "
                "Return the n node temperatures.")


def test_b_correct_profile_passes():
    code = ("def f(L, n, T_left, T_right):\n"
            "    return [T_left + (T_right - T_left)*i/(n-1) for i in range(n)]\n")
    report = q.check_problem(type_b(PROFILE_TEXT, code, "f", [[1.0, 5, 100.0, 0.0]]))
    assert report.passed, report.summary()


def test_b_temperature_out_of_bounds_fails():
    code = ("def f(L, n, T_left, T_right):\n"
            "    return [T_left - 50*i*L for i in range(n)]\n")
    r = q.check_b_temperature_bounds(type_b(PROFILE_TEXT, code, "f", [[1.0, 5, 100.0, 0.0]]))
    assert not r.passed


def test_b_bounds_skipped_with_heat_source():
    text = PROFILE_TEXT.replace("conduction.", "conduction with heat generation q_gen.")
    code = "def f(L, n, T_left, T_right):\n    return [500.0]*n\n"
    assert q.check_b_temperature_bounds(type_b(text, code, "f", [[1.0, 3, 100.0, 0.0]])).skipped


def test_b_non_finite_output_fails():
    code = "def f(L, n, T_left, T_right):\n    return [float('inf')]*n\n"
    p = type_b(PROFILE_TEXT, "def f(L, n, T_left, T_right):\n    return [1e400]*n\n", "f",
               [[1.0, 2, 0.0, 1.0]])
    assert not q.check_b_outputs_finite(p).passed


def test_b_mostly_unused_args_fail():
    code = "def h(T_wall, T_amb, h, L, k):\n    return k / L\n"
    r = q.check_b_all_args_used(type_b("Return h.", code, "h", [[20, 10, 100, 0.1, 0.5]]))
    assert not r.passed


def test_b_one_unused_arg_is_tolerated():
    code = "def f(L, n, T_left, T_right):\n    return [T_left, T_right][:n]\n"
    r = q.check_b_all_args_used(type_b(PROFILE_TEXT, code, "f", [[1.0, 2, 1.0, 2.0]]))
    assert r.passed and "L" in r.detail


def test_b_ftcs_stability():
    text = "Write transient(L, n, alpha, dt) ... Return the final temperatures."
    code = "def transient(L, n, alpha, dt):\n    return [L, n, alpha, dt]\n"
    unstable = type_b(text, code, "transient", [[1.0, 11, 1e-4, 100.0]])   # r = 1.0
    stable = type_b(text, code, "transient", [[1.0, 11, 1e-4, 10.0]])      # r = 0.1
    assert not q.check_b_explicit_stability(unstable).passed
    assert q.check_b_explicit_stability(stable).passed


def test_b_forward_euler_stability():
    code = "def cool(T0, T_inf, tau, t_end, dt):\n    return [T0, T_inf, tau, t_end, dt]\n"
    p = type_b("Return the temperature history.", code, "cool", [[100, 20, 1.0, 10, 2.0]])
    assert not q.check_b_explicit_stability(p).passed


###############################################################################
# Regression against the hand-labelled audit
###############################################################################

def _audit():
    labels = json.loads((FIXTURES / "audit_labels.json").read_text())["labels"]
    problems = json.loads((FIXTURES / "generated_examples_2026-11-02.json").read_text())["seeds"]
    return problems, labels


def test_audit_no_correct_problem_rejected():
    problems, labels = _audit()
    wrongly_rejected = [p["id"] for p in problems
                        if labels[p["id"]]["verdict"] != "wrong" and not q.check_problem(p).passed]
    assert wrongly_rejected == []


def test_audit_catches_most_wrong_problems():
    # 48 of 72 wrong problems were caught when these checks were written.
    # Raise this floor when the checks improve; never lower it silently.
    problems, labels = _audit()
    caught = sum(1 for p in problems
                 if labels[p["id"]]["verdict"] == "wrong" and not q.check_problem(p).passed)
    assert caught >= 48


# Seeds that the deterministic checks flag, with the reason. These are the
# author's own problems, so each one is either a seed to fix or a known limit.
KNOWN_SEED_FLAGS = {
    "cond-A-04": "uses the precomputed resistance sum 0.137",
    "cond-A-07": "text says the steel layer is 1.5 mm but the formula uses 1*1e-3",
    "cond-A-08": "the instruction never asks for a quantity",
    "conv-A-09": "uses the precomputed characteristic length 0.075",
    "conv-A-11": "uses precomputed intermediate values (Re = 206800, 0.04625, 0.6667)",
}


@pytest.mark.skipif(not SEEDS_PATH.exists(), reason="seed file not built")
def test_seeds_pass_except_known():
    seeds = json.loads(SEEDS_PATH.read_text())["seeds"]
    flagged = {s["id"] for s in seeds if not q.check_problem(s).passed}
    assert flagged <= set(KNOWN_SEED_FLAGS), f"new seed failures: {flagged - set(KNOWN_SEED_FLAGS)}"


###############################################################################
# Independent re-solve (fake second model)
###############################################################################

def fake_chat(content: dict):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps(content)}}
    chat.calls = calls
    return chat


def test_resolve_a_agreement_passes():
    p = type_a(WALL, "0.6*3*(25-15)/0.2", "W")
    chat = fake_chat({"reasoning": "Q = kA dT / L", "check_expression": "0.6*3*10/0.2"})
    r = q.resolve_type_a(p, "judge", chat)
    assert r.passed
    assert chat.calls[0]["model"] == "judge"
    assert "0.6*3*(25-15)/0.2" not in chat.calls[0]["messages"][0]["content"]  # judge sees text only


def test_resolve_a_disagreement_fails():
    p = type_a(WALL, "0.6*3*(25-15)/0.2", "W")
    r = q.resolve_type_a(p, "judge", fake_chat({"reasoning": "x", "check_expression": "0.6*3*10"}))
    assert not r.passed


def test_resolve_a_unsolvable_fails():
    p = type_a(WALL, "0.6*3*(25-15)/0.2", "W")
    r = q.resolve_type_a(p, "judge", fake_chat({"reasoning": "missing L", "check_expression": "None"}))
    assert not r.passed and "unsolvable" in r.detail


def test_resolve_a_model_error_fails():
    def broken(**kwargs):
        raise RuntimeError("server down")
    r = q.resolve_type_a(type_a(WALL, "1", "W"), "judge", broken)
    assert not r.passed and "failed" in r.detail


def test_resolve_b_agreement_and_disagreement():
    code = ("def f(L, n, T_left, T_right):\n"
            "    return [T_left + (T_right - T_left)*i/(n-1) for i in range(n)]\n")
    p = type_b(PROFILE_TEXT, code, "f", [[1.0, 5, 100.0, 0.0]])
    same = "def f(L, n, T_left, T_right):\n    return [T_left - (T_left - T_right)*i/(n-1) for i in range(n)]\n"
    other = "def f(L, n, T_left, T_right):\n    return [T_left]*n\n"
    assert q.resolve_type_b(p, "judge", fake_chat({"reference_solution": same})).passed
    assert not q.resolve_type_b(p, "judge", fake_chat({"reference_solution": other})).passed


def test_assess_runs_resolve_only_after_deterministic_pass():
    good = type_a(WALL, "0.6*3*(25-15)/0.2", "W")
    bad = type_a(WALL, "0.6*3*(15-25)/0.2", "W")  # negative
    chat = fake_chat({"reasoning": "x", "check_expression": "0.6*3*10/0.2"})
    assert q.assess_problem(good, "judge", chat).passed
    assert len(chat.calls) == 1
    assert not q.assess_problem(bad, "judge", chat).passed
    assert len(chat.calls) == 1  # the judge was not called for the failing problem


def test_assess_without_judge_is_deterministic_only():
    report = q.assess_problem(type_a(WALL, "0.6*3*(25-15)/0.2", "W"))
    assert "resolve_agreement" not in [r.name for r in report.results]
