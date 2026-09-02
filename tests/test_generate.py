"""
Tests for generate.py

These test the post-processing logic of generation not the actual LLM call. 
`ollama.chat` is mocked to return controlled responses, so the tests
run offline and deterministically.

What is (and isn't) tested here:
  * Tested: computing answer_numeric from check_expression (Type A); building
    test_cases by running the reference (Type B); graceful None-return on bad input.
  * Not tested: the real ollama.chat call, main()/CLI, and file writing — these are
    I/O/orchestration and are intentionally left to manual or integration testing.

"""

import json
import sys
import types

import pytest


class _MockOllama:
    """A stand-in for the ollama module. Set `.next_response` before each call."""
    next_response = None

    @staticmethod
    def chat(**kwargs):
        # Return the queued response as if it came from the model.
        return {"message": {"content": json.dumps(_MockOllama.next_response)}}


sys.modules["ollama"] = types.SimpleNamespace(chat=_MockOllama.chat)


from syntheticheatransfer import generate as g   


###############################################################################
# Type A
###############################################################################

SEED_A = {
    "id": "cond-A-01",
    "type": "A",
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