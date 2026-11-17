"""
Tests for generate.py

These test the post-processing logic of generation not the actual LLM call. 
`ollama.chat` is mocked to return controlled responses, so the tests
run offline and deterministically.

What is (and isn't) tested here:
  * Tested: computing answer_numeric from check_expression (Type A); building
    test_cases by running the reference (Type B); graceful None-return on bad input;
    provenance fields (source seed, topic, subtopic, generator model) on generated items.
  * Also tested: generate_dataset keeps problems that pass the quality checks and
    returns the rest as rejected (quality.py itself is tested in test_quality.py);
    the repair loop sends failure reasons back, stops when a draft passes or the
    limit is reached, and never shows the generator the judge model's answer.
  * Not tested: the real ollama.chat call, main()/CLI, and file writing — these are
    I/O/orchestration and are intentionally left to manual or integration testing.

"""

import json
import sys
import types

import pytest


class _MockOllama:
    """
    A stand-in for the ollama module.
      * Set `.next_response` for tests that make a single call (returned every time).
      * Or set `.responses` to a list for tests that make several calls (repair
        loop, judge model): each call returns the next item. An item that is a
        string is returned as raw content (e.g. invalid JSON).
    Every call's keyword arguments are recorded in `.calls`.
    """
    next_response = None
    responses: list = []
    calls: list = []

    @staticmethod
    def chat(**kwargs):
        _MockOllama.calls.append(kwargs)
        if _MockOllama.responses:
            item = _MockOllama.responses.pop(0)
        else:
            item = _MockOllama.next_response
        content = item if isinstance(item, str) else json.dumps(item)
        return {"message": {"content": content}}


@pytest.fixture(autouse=True)
def _reset_mock():
    _MockOllama.responses = []
    _MockOllama.calls = []
    yield


sys.modules["ollama"] = types.SimpleNamespace(chat=_MockOllama.chat)


from syntheticheatransfer import generate as g   


###############################################################################
# Type A
###############################################################################

SEED_A = {
    "id": "cond-A-01",
    "type": "A",
    "topic": "conduction",
    "subtopic": "plane wall, Fourier's law",
    "instruction": "example",
    "check_expression": "1.2*(300-50)/0.15",
    "answer_units": "W/m^2",
}


def test_type_a_valid_computes_answer():
    # Model returns a valid expression; answer_numeric must be computed from it.
    _MockOllama.next_response = {
        "instruction": "A wall ...",
        "reasoning": "q = k*dT/L",
        "answer_units": "W/m^2",
        "check_expression": "2.0*(100-20)/0.1", 
    }
    problem = g.generate_one_a(SEED_A, "test-model")
    assert problem is not None
    assert problem["type"] == "A"
    assert problem["answer_numeric"] == pytest.approx(1600.0)
    assert problem["provenance"] == "llm_generated_verified"
    assert problem["source_seed_id"] == "cond-A-01"


def test_type_a_records_provenance():
    # Generated items must record the seed's topic/subtopic and the generator model.
    _MockOllama.next_response = {
        "instruction": "A wall ...",
        "reasoning": "q = k*dT/L",
        "answer_units": "W/m^2",
        "check_expression": "2.0*(100-20)/0.1",
    }
    problem = g.generate_one_a(SEED_A, "test-model")
    assert problem is not None
    assert problem["generator_model"] == "test-model"
    assert problem["topic"] == "conduction"
    assert problem["subtopic"] == "plane wall, Fourier's law"


def test_type_a_uses_math_and_numpy():
    _MockOllama.next_response = {
        "instruction": "A pipe ...",
        "reasoning": "cylindrical",
        "answer_units": "W",
        "check_expression": "2*math.pi*15*1*(100-40)/math.log(0.03/0.02)",
    }
    problem = g.generate_one_a(SEED_A, "test-model")
    assert problem is not None
    assert problem["answer_numeric"] == pytest.approx(13946.6, rel=1e-3)


def test_type_a_bad_expression_returns_none():
    # A malformed check_expression must be caught -> None, not a crash.
    _MockOllama.next_response = {
        "instruction": "bad",
        "reasoning": "r",
        "answer_units": "W",
        "check_expression": "this is not valid python !!!",
    }
    assert g.generate_one_a(SEED_A, "test-model") is None


def test_type_a_disallowed_name_returns_none():
    # Bare 'pi' (not math.pi) is not in the namespace -> NameError -> None.
    _MockOllama.next_response = {
        "instruction": "x",
        "reasoning": "r",
        "answer_units": "W",
        "check_expression": "some_undefined_name * 2",
    }
    assert g.generate_one_a(SEED_A, "test-model") is None


###############################################################################
# Type B
###############################################################################

SEED_B = {
    "id": "cond-B-01",
    "type": "B",
    "topic": "conduction",
    "subtopic": "1-D steady finite difference",
    "instruction": "example",
    "function_name": "solve",
    "reference_solution": "def solve():\n    return 0\n",
}


def test_type_b_valid_builds_test_cases():
    # A working reference + test args -> expected computed by running the reference.
    _MockOllama.next_response = {
        "instruction": "double each",
        "function_name": "double_all",
        "reference_solution": "def double_all(xs):\n    return [x*2 for x in xs]\n",
        "test_args": [[[1, 2, 3]], [[0, 5]]],  
    }
    problem = g.generate_one_b(SEED_B, "test-model")
    assert problem is not None
    assert problem["type"] == "B"
    # expected computed from the reference
    assert problem["test_cases"][0]["expected"] == [2, 4, 6]
    assert problem["test_cases"][1]["expected"] == [0, 10]


def test_type_b_records_provenance_and_drops_test_args():
    # Provenance is recorded, and the raw test_args are not kept because the same
    # inputs are stored in test_cases[i]["args"].
    _MockOllama.next_response = {
        "instruction": "double each",
        "function_name": "double_all",
        "reference_solution": "def double_all(xs):\n    return [x*2 for x in xs]\n",
        "test_args": [[[1, 2, 3]], [[0, 5]]],
    }
    problem = g.generate_one_b(SEED_B, "test-model")
    assert problem is not None
    assert problem["generator_model"] == "test-model"
    assert problem["topic"] == "conduction"
    assert problem["subtopic"] == "1-D steady finite difference"
    assert "test_args" not in problem
    assert [c["args"] for c in problem["test_cases"]] == [[[1, 2, 3]], [[0, 5]]]


def test_type_b_runtime_error_returns_none():
    # Code that raises on its own test args must be dropped (like ZeroDivisionError).
    _MockOllama.next_response = {
        "instruction": "divide",
        "function_name": "div",
        "reference_solution": "def div(a, b):\n    return a / b\n",
        "test_args": [[1, 0]],   # division by zero -> should be caught -> None
    }
    assert g.generate_one_b(SEED_B, "test-model") is None


def test_type_b_wrong_function_name_returns_none():
    # Reference defines a different function than function_name -> load fails -> None.
    _MockOllama.next_response = {
        "instruction": "mismatch",
        "function_name": "expected_name",
        "reference_solution": "def other_name():\n    return 1\n",
        "test_args": [[]],
    }
    assert g.generate_one_b(SEED_B, "test-model") is None


###############################################################################
# generate_dataset: verification + quality checks
###############################################################################

def _write_seed_file(tmp_path, seed):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"seeds": [seed]}))
    return path


def test_dataset_keeps_problem_that_passes_quality_checks(tmp_path):
    _MockOllama.next_response = {
        "instruction": ("A plane wall 0.2 m thick with area 3 m^2 has k = 0.6 W/m*K. Its "
                        "surfaces are at 25 C and 15 C. Find the heat transfer rate in W."),
        "reasoning": "Q = k*A*dT/L",
        "answer_units": "W",
        "check_expression": "0.6*3*(25-15)/0.2",
    }
    kept, stats, rejected = g.generate_dataset(_write_seed_file(tmp_path, SEED_A), "m", 1)
    assert len(kept) == 1 and rejected == []
    assert kept[0]["quality_checks"] == "passed"
    assert stats["failed_quality"] == 0 and stats["quality_checks"] == "deterministic"


def test_dataset_rejects_problem_that_fails_quality_checks(tmp_path):
    # Self-consistent (passes verification) but the heat rate is negative.
    _MockOllama.next_response = {
        "instruction": ("A plane wall 0.2 m thick with area 3 m^2 has k = 0.6 W/m*K. Its "
                        "surfaces are at 25 C and 15 C. Find the heat transfer rate in W."),
        "reasoning": "Q = k*A*dT/L",
        "answer_units": "W",
        "check_expression": "0.6*3*(15-25)/0.2",
    }
    kept, stats, rejected = g.generate_dataset(_write_seed_file(tmp_path, SEED_A), "m", 1)
    assert kept == [] and len(rejected) == 1
    assert "a_positive_magnitude" in rejected[0]["quality"]["failed_checks"]
    assert stats["failed_quality"] == 1
    assert stats["failed_by_check"]["a_positive_magnitude"] == 1


def test_dataset_quality_checks_can_be_turned_off(tmp_path):
    _MockOllama.next_response = {
        "instruction": "A wall ... Find the heat transfer rate in W.",
        "reasoning": "r",
        "answer_units": "W",
        "check_expression": "0.6*3*(15-25)/0.2",
    }
    kept, stats, rejected = g.generate_dataset(_write_seed_file(tmp_path, SEED_A), "m", 1,
                                               quality_checks=False)
    assert len(kept) == 1 and stats["quality_checks"] == "off"


def test_dataset_skips_unknown_seed_type(tmp_path):
    # A seed with an unknown type is skipped and not counted as an attempt.
    kept, stats, rejected = g.generate_dataset(
        _write_seed_file(tmp_path, {"id": "x-01", "type": "Z"}), "m", 2)
    assert kept == [] and rejected == []
    assert stats["attempted"] == 0


###############################################################################
# Repair loop
###############################################################################

WALL_TEXT = ("A plane wall 0.2 m thick with area 3 m^2 has k = 0.6 W/m*K. Its "
             "surfaces are at 25 C and 15 C. Find the heat transfer rate in W.")
GOOD_A = {"instruction": WALL_TEXT, "reasoning": "Q = k*A*dT/L", "answer_units": "W",
          "check_expression": "0.6*3*(25-15)/0.2"}
NEGATIVE_A = dict(GOOD_A, check_expression="0.6*3*(15-25)/0.2")


def _prompt(call_index):
    return _MockOllama.calls[call_index]["messages"][0]["content"]


def test_no_repair_by_default_makes_one_call():
    _MockOllama.responses = [NEGATIVE_A, GOOD_A]
    status, problem, history = g.generate_with_repair(SEED_A, "m")
    assert status == "failed_quality"
    assert len(_MockOllama.calls) == 1 and len(history) == 1
    assert "repair_rounds" not in problem


def test_repair_fixes_failing_draft():
    _MockOllama.responses = [NEGATIVE_A, GOOD_A]
    status, problem, history = g.generate_with_repair(SEED_A, "m", max_repairs=2)
    assert status == "kept"
    assert len(_MockOllama.calls) == 2           # stopped as soon as it passed
    assert problem["answer_numeric"] == pytest.approx(90.0)
    assert problem["repair_rounds"] == 1
    assert [h["status"] for h in history] == ["failed_quality", "kept"]
    # The repair prompt shows the draft and the reason it failed.
    repair_prompt = _prompt(1)
    assert "0.6*3*(15-25)/0.2" in repair_prompt
    assert "a_positive_magnitude" in repair_prompt
    assert _MockOllama.calls[1]["options"]["temperature"] == g.REPAIR_TEMPERATURE


def test_repair_gives_up_after_max_rounds():
    _MockOllama.responses = [NEGATIVE_A, NEGATIVE_A, NEGATIVE_A]
    status, problem, history = g.generate_with_repair(SEED_A, "m", max_repairs=2)
    assert status == "failed_quality"
    assert len(_MockOllama.calls) == 3 and len(history) == 3
    assert problem["repair_rounds"] == 2


def test_repair_fixes_expression_that_does_not_evaluate():
    _MockOllama.responses = [dict(GOOD_A, check_expression="0.6*3*(25-15)/L"), GOOD_A]
    status, problem, history = g.generate_with_repair(SEED_A, "m", max_repairs=1)
    assert status == "kept"
    assert history[0]["status"] == "unparsed"
    assert "NameError" in _prompt(1)


def test_repair_fixes_type_b_runtime_error():
    bad = {"instruction": "Return the n node temperatures.", "function_name": "f",
           "reference_solution": "def f(n, T_left, T_right):\n    return [T_left/(n-n)]\n",
           "test_args": [[3, 100.0, 0.0]]}
    good = dict(bad, reference_solution=(
        "def f(n, T_left, T_right):\n"
        "    return [T_left + (T_right - T_left)*i/(n-1) for i in range(n)]\n"))
    _MockOllama.responses = [bad, good]
    status, problem, history = g.generate_with_repair(SEED_B, "m", max_repairs=1)
    assert status == "kept"
    assert "ZeroDivisionError" in _prompt(1)
    assert "test_args" not in problem and problem["test_cases"][0]["args"] == [3, 100.0, 0.0]


def test_repair_never_reveals_the_judge_answer():
    # Round 0: generator draft, then judge disagrees with a distinctive expression.
    # Round 1: revised draft, then judge agrees.
    judge_wrong = {"reasoning": "x", "check_expression": "0.6*3*(25-15)/0.2*777"}
    judge_right = {"reasoning": "x", "check_expression": "0.6*3*10/0.2"}
    _MockOllama.responses = [GOOD_A, judge_wrong, GOOD_A, judge_right]
    status, problem, history = g.generate_with_repair(SEED_A, "m", max_repairs=1,
                                                      judge_model="judge")
    assert status == "kept"
    assert [c["model"] for c in _MockOllama.calls] == ["m", "judge", "m", "judge"]
    repair_prompt = _prompt(2)
    assert "777" not in repair_prompt
    assert g.RESOLVE_FEEDBACK in repair_prompt


def test_repair_stops_if_model_returns_invalid_json():
    _MockOllama.responses = [NEGATIVE_A, "not json"]
    status, problem, history = g.generate_with_repair(SEED_A, "m", max_repairs=3)
    assert status == "failed_quality"            # result of the last good round
    assert len(_MockOllama.calls) == 2
    assert history[-1]["status"] == "unparsed"


def test_dataset_counts_repairs(tmp_path):
    _MockOllama.responses = [NEGATIVE_A, GOOD_A]
    kept, stats, rejected = g.generate_dataset(_write_seed_file(tmp_path, SEED_A), "m", 1,
                                               max_repairs=1)
    assert len(kept) == 1 and rejected == []
    assert stats["kept_after_repair"] == 1 and stats["repair_rounds"] == 1
    assert stats["max_repairs"] == 1
    assert kept[0]["id"] == "gen-cond-A-01-1"


def test_quality_feedback_lists_each_failed_check():
    from syntheticheatransfer import quality as q
    report = q.QualityReport("x", [q.CheckResult("a_dimensions", False, "formula gives W m"),
                                   q.CheckResult("a_precedence", True),
                                   q.CheckResult("resolve_agreement", False, "judge said 42")])
    lines = g.quality_feedback(report)
    assert lines[0] == "a_dimensions: formula gives W m"
    assert lines[1] == g.RESOLVE_FEEDBACK and "42" not in " ".join(lines)


###############################################################################
# Prompts (pure functions)
###############################################################################

def test_prompt_a_includes_example_fields():
    prompt = g.build_prompt_a(SEED_A)
    assert SEED_A["check_expression"] in prompt
    assert "check_expression" in prompt


def test_prompt_b_includes_example_fields():
    prompt = g.build_prompt_b(SEED_B)
    assert SEED_B["function_name"] in prompt
    assert "reference_solution" in prompt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))