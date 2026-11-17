"""
Synthetic problem generation using Ollama with structured output,
deterministic verification and semantic quality checks.

Design principles, given the limits of Ollama and other small local LLMs:
  * The model constructs problems: scenario + formula/code.
  * The model is not required to do any arithmetic to supply the final numeric
    answer (Type A) or the expected test values (Type B). The code computes those:
      - Type A: answer_numeric = eval(check_expression)
      - Type B: expected = reference_solution(*args) for each test case
    This makes every kept problem self-consistent by construction.
  * Ollama's `format` (JSON schema) constrains output so it parses reliably.
  * Every generated problem goes through two filters, and only problems that
    pass both are kept:
      1. deterministic verification (verification.py), then
      2. semantic quality checks (quality.py): deterministic sanity checks, plus
         an independent re-solve by a second model when --judge-model is given.
    Problems that pass verification but fail a quality check are written to a
    separate file with the reasons, for analysis.

Verification semantics:
  * Type A "passed" = check_expression is valid/evaluable and the answer matches it.
  * Type B "passed" = reference_solution runs deterministically on its test args.
  Neither guarantees that the formula/code matches the wording or the physics;
  that is what the quality checks in quality.py are for.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import ollama
from syntheticheatransfer import verification as v
from syntheticheatransfer import quality as q


###############################################################################
# Settings
###############################################################################

TYPE_A_TEMPERATURE = 0.6
TYPE_B_TEMPERATURE = 0.3
TYPE_A_TOLERANCE_REL = 0.02
TYPE_B_TOLERANCE_REL = 1e-6


###############################################################################
# Shared helpers
###############################################################################

def _add_provenance(obj: dict, example: dict, model: str) -> None:
    """
    Record where a generated problem came from: the seed it was modeled on
    (id, topic, subtopic) and the model that wrote it. Shared by Type A and
    Type B so the two can never drift apart.
    """
    obj["provenance"] = "llm_generated_verified"
    obj["source_seed_id"] = example.get("id")
    obj["topic"] = example.get("topic")
    obj["subtopic"] = example.get("subtopic")
    obj["generator_model"] = model


def _chat_json(model: str, prompt: str, schema: dict, temperature: float) -> dict:
    """One structured-output call to Ollama, parsed into a dict."""
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature},
    )
    return json.loads(resp["message"]["content"])


###############################################################################
# Type A: numeric-answer problems
###############################################################################

TYPE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "reasoning": {"type": "string"},
        "answer_units": {"type": "string"},
        "check_expression": {"type": "string"},
    },
    "required": ["instruction", "reasoning", "answer_units", "check_expression"],
}


def build_prompt_a(example: dict) -> str:
    return f"""You are creating a heat-transfer problem for a dataset. You are given one
example problem. Create ONE NEW problem in the SAME style and difficulty.

Rules:
- It must be a single-numeric-answer heat-transfer problem (like the example).
- Change the scenario and the numbers; do NOT copy the example.
- Keep it solvable with a short formula. Do NOT create open-ended, symbolic, or
  multi-part problems.
- In 'check_expression', use Python's math module and/or numpy explicitly: write 'math.pi',
  'math.sqrt(...)', 'math.log(...)', 'np.pi', 'np.linalg.inv(...)', 'np.array(...)'. Do NOT use
  bare 'pi', the symbol π, or any name not from the math or numpy modules. Use only numbers,
  math.*, and np.* functions.
- 'check_expression' must contain the ACTUAL numbers from your problem and must
  compute the numeric answer. The answer will be computed from this expression.
- Every number used in 'check_expression' must be stated in the problem text,
  with its units. Do NOT assume material properties, coefficients, temperatures,
  or dimensions that the problem does not give.
- Use parentheses around every temperature difference, e.g. h*A*(T_s - T_inf).

Example problem:
  instruction: {example.get('instruction', '')}
  check_expression: {example.get('check_expression', '')}
  answer_units: {example.get('answer_units', '')}

Now produce a new problem as JSON with fields: instruction, reasoning,
answer_units, check_expression."""


def generate_one_a(example: dict, model: str) -> dict | None:
    """
    Ask the model for one Type A problem modeled on `example`. The answer is
    computed from the model's check_expression, never taken from the model.
    Returns None if the call, the parse or the evaluation fails.
    """
    try:
        obj = _chat_json(model, build_prompt_a(example), TYPE_A_SCHEMA, TYPE_A_TEMPERATURE)
        obj["answer_numeric"] = float(eval(obj["check_expression"], v.expression_namespace()))
    except Exception as e:
        print(f"  type-A generation/parse failed: {e!r}")
        return None

    obj["type"] = "A"
    _add_provenance(obj, example, model)
    obj["answer_tolerance_rel"] = TYPE_A_TOLERANCE_REL
    return obj


###############################################################################
# Type B: coding problems
###############################################################################

TYPE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "function_name": {"type": "string"},
        "reference_solution": {"type": "string"},
        "test_args": {"type": "array", "items": {"type": "array"}},
    },
    "required": ["instruction", "function_name", "reference_solution", "test_args"],
}


def build_prompt_b(example: dict) -> str:
    return f"""Create ONE NEW heat-transfer coding problem in the same style as the example.

Rules:
- The problem asks for a Python function that performs a heat-transfer computation.
- 'reference_solution' must be a COMPLETE, CORRECT Python function definition (as a
  string), named exactly by 'function_name'. It may import and use numpy and math.
- 'test_args' is a list of argument lists; each inner list holds the positional
  arguments for one call to the function. Choose simple, valid inputs.
- Make it concrete and computable. Do NOT make it open-ended or symbolic.
- Change the scenario and numbers; do NOT copy the example.
- Write 'reference_solution' as valid, properly-indented Python using real newlines
  (\\n) between lines and 4-space indentation for the function body. Do NOT put the
  whole function on one line. Do NOT use semicolons to separate statements. It must
  be code that runs as-is with exec().
- Choose 'test_args' that are valid and safe for function evaluation: no zero
  denominators, no empty arrays, no out-of-range indices. Each test call must
  run without error.
- Make sure the number and type of arguments in each test_args entry exactly match
  the function's parameters.
- Prefer simple, small, clearly-valid inputs (e.g. positive numbers, small arrays).

Example:
  instruction: {example.get('instruction', '')}
  function_name: {example.get('function_name', '')}
  reference_solution: {example.get('reference_solution', '')}

Produce JSON with fields: instruction, function_name, reference_solution, test_args."""


def generate_one_b(example: dict, model: str) -> dict | None:
    """
    Ask the model for one Type B problem modeled on `example`. Expected values
    are computed by running the model's reference_solution on its test_args.
    Returns None if the call, the parse, loading or any test call fails.
    """
    try:
        obj = _chat_json(model, build_prompt_b(example), TYPE_B_SCHEMA, TYPE_B_TEMPERATURE)
        fn = v.load_solution(obj["reference_solution"], obj["function_name"])
        test_cases = []
        for i, args in enumerate(obj["test_args"]):
            expected = fn(*list(args))
            test_cases.append({
                "name": f"case_{i + 1}",
                "args": list(args),
                "expected": expected,
                "tol_rel": TYPE_B_TOLERANCE_REL,
            })
        if not test_cases:
            print("  type-B: no test_args provided; dropping.")
            return None
        obj["test_cases"] = test_cases
        obj.pop("test_args", None)  # the inputs now live in test_cases[i]["args"]
    except Exception as e:
        print(f"  type-B generation/parse failed: {e!r}")
        return None

    obj["type"] = "B"
    _add_provenance(obj, example, model)
    return obj


GENERATORS = {"A": generate_one_a, "B": generate_one_b}


###############################################################################
# Generation
###############################################################################

def generate_dataset(
    seeds_path: Path,
    model: str,
    per_seed: int,
    quality_checks: bool = True,
    judge_model: str | None = None,
) -> tuple[list, dict, list]:
    """
    Generate `per_seed` problems per seed and keep only those that pass
    deterministic verification and (unless quality_checks=False) the semantic
    quality checks. If `judge_model` is given, problems that pass the
    deterministic quality checks are also re-solved independently by that model.

    Returns (kept, stats, rejected). `rejected` holds problems that passed
    verification but failed a quality check, each with a 'quality' summary.
    """
    seeds = json.loads(Path(seeds_path).read_text(encoding="utf-8"))["seeds"]

    kept, rejected = [], []
    attempted = passed = failed = unparsed = failed_quality = 0
    failed_by_check: Counter = Counter()

    for seed in seeds:
        stype = seed.get("type")
        generate_one = GENERATORS.get(stype)
        if generate_one is None:
            print(f"  skipping seed {seed.get('id')}: unknown type {stype!r}")
            continue

        for _ in range(per_seed):
            attempted += 1
            problem = generate_one(seed, model)
            if problem is None:
                unparsed += 1
                continue

            problem["id"] = f"gen-{seed['id']}-{attempted}"
            result = v.verify_problem(problem)
            if not result.passed:
                failed += 1
                print(f"  FAIL verification {problem['id']} ({stype}): {result.detail}")
                continue
            problem["verification"] = "passed"

            if quality_checks:
                report = q.assess_problem(problem, judge_model=judge_model)
                if not report.passed:
                    failed_quality += 1
                    failed_by_check.update(report.failed_checks)
                    problem["quality"] = report.summary()
                    rejected.append(problem)
                    print(f"  FAIL quality {problem['id']} ({stype}): "
                          f"{', '.join(report.failed_checks)}")
                    continue
                problem["quality_checks"] = "passed"
                if judge_model:
                    problem["judge_model"] = judge_model

            kept.append(problem)
            passed += 1
            print(f"  PASS {problem['id']} ({stype})")

    stats = {
        "attempted": attempted,
        "passed": passed,
        "failed_verification": failed,
        "failed_quality": failed_quality,
        "failed_by_check": dict(failed_by_check),
        "unparsed": unparsed,
        "quality_checks": ("off" if not quality_checks
                           else "deterministic + independent re-solve" if judge_model
                           else "deterministic"),
    }
    return kept, stats, rejected


###############################################################################
# Command line
###############################################################################

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data"

SEEDS_PATH = DATA_DIR / "seeds" / "heat_transfer_seeds.json"
OUTPUT_PATH = DATA_DIR / "generated" / "generated_examples.json"
REJECTED_PATH = DATA_DIR / "generated" / "rejected_examples.json"


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON, creating the parent folder if needed (only where we write)."""
    v.create_output_folder(path.parent.parent, path.parent.name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct-Ollama verifiable problem generation.")
    ap.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--per-seed", type=int, default=2,
                    help="How many new problems to generate per seed.")
    ap.add_argument("--judge-model", default=None,
                    help="Second model for the independent re-solve (e.g. qwen2.5:latest). "
                         "Should differ from --model. Omit to run deterministic checks only.")
    ap.add_argument("--no-quality-checks", action="store_true",
                    help="Keep everything that passes verification (not recommended).")
    ap.add_argument("--rejected-out", type=Path, default=REJECTED_PATH,
                    help="Where to write problems that failed a quality check.")
    args = ap.parse_args()
    if args.judge_model and args.judge_model == args.model:
        print("WARNING: --judge-model equals --model; the re-solve is not independent.")

    print(f"Model: {args.model} | judge: {args.judge_model} | seeds: {args.seeds}")
    kept, stats, rejected = generate_dataset(
        args.seeds, args.model, args.per_seed,
        quality_checks=not args.no_quality_checks, judge_model=args.judge_model,
    )
    print(f"\nStats: {stats}")
    print("Kept by type:", dict(Counter(p["type"] for p in kept)))

    _write_json(args.out, {
        "domain": "heat_transfer",
        "description": ("LLM-generated problems that passed deterministic verification "
                        f"and quality checks ({stats['quality_checks']})."),
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "generation_stats": stats,
        "seeds": kept,
    })
    print(f"Wrote {len(kept)} examples to {args.out}")

    if rejected:
        _write_json(args.rejected_out, {
            "description": "Passed verification, failed a quality check (see 'quality').",
            "rejected": rejected,
        })
        print(f"Wrote {len(rejected)} rejected examples to {args.rejected_out}")


if __name__ == "__main__":
    main()
