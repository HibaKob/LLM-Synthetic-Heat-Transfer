"""
Synthetic problem generation using Ollama with structured output,
deterministic verification, semantic quality checks and an optional
repair loop.

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
  * Repair loop (--max-repairs N, off by default): when a draft fails to
    evaluate, fails verification, or fails a quality check, the failure reasons
    are sent back to the generator, which revises the draft, up to N times.
    Safeguards: every revision is re-checked from scratch by the same filters;
    the generator is told only that the independent solver disagreed, never
    what the solver answered, so the re-solve stays independent.
  * Problems that still fail are written to a separate file with the reasons
    and the repair history, for analysis.

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
REPAIR_TEMPERATURE = 0.3          # revisions should change what was flagged, not everything
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


def _feedback_lines(lines: list) -> str:
    return "\n".join(f"- {line}" for line in lines)


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

# Shared by the generation prompt and the repair prompt.
RULES_A = """- It must be a single-numeric-answer heat-transfer problem.
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
- Use parentheses around every temperature difference, e.g. h*A*(T_s - T_inf)."""


def build_prompt_a(example: dict) -> str:
    return f"""You are creating a heat-transfer problem for a dataset. You are given one
example problem. Create ONE NEW problem in the SAME style and difficulty.

Rules:
- Change the scenario and the numbers; do NOT copy the example.
{RULES_A}

Example problem:
  instruction: {example.get('instruction', '')}
  check_expression: {example.get('check_expression', '')}
  answer_units: {example.get('answer_units', '')}

Now produce a new problem as JSON with fields: instruction, reasoning,
answer_units, check_expression."""


def build_repair_prompt_a(draft: dict, feedback: list) -> str:
    return f"""You wrote the heat-transfer problem below for a dataset. Automatic checks
found the problems listed under "Issues found". Revise it so every issue is fixed.

How to revise:
- Fix the physics and the formula. Do NOT just change the units label or the
  wording to hide a mistake: the checks compare the formula with the numbers and
  units stated in the problem text.
- If the problem is missing information it needs, add it to the problem text.
- Keep the same topic. You may change numbers, wording, formula and units.

Rules:
{RULES_A}

Your problem:
  instruction: {draft.get('instruction', '')}
  check_expression: {draft.get('check_expression', '')}
  answer_units: {draft.get('answer_units', '')}

Issues found:
{_feedback_lines(feedback)}

Produce the revised problem as JSON with fields: instruction, reasoning,
answer_units, check_expression."""


def _finalize_a(raw: dict, example: dict, model: str) -> dict:
    """
    Turn the model's JSON into a Type A problem: compute answer_numeric from the
    check_expression and add type, provenance and tolerance. Raises on failure.
    """
    obj = dict(raw)
    obj["answer_numeric"] = float(eval(obj["check_expression"], v.expression_namespace()))
    obj["type"] = "A"
    _add_provenance(obj, example, model)
    obj["answer_tolerance_rel"] = TYPE_A_TOLERANCE_REL
    return obj


def generate_one_a(example: dict, model: str) -> dict | None:
    """
    Ask the model for one Type A problem modeled on `example`. The answer is
    computed from the model's check_expression, never taken from the model.
    Returns None if the call, the parse or the evaluation fails.
    """
    try:
        raw = _chat_json(model, build_prompt_a(example), TYPE_A_SCHEMA, TYPE_A_TEMPERATURE)
        return _finalize_a(raw, example, model)
    except Exception as e:
        print(f"  type-A generation/parse failed: {e!r}")
        return None


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

# Shared by the generation prompt and the repair prompt.
RULES_B = """- The problem asks for a Python function that performs a heat-transfer computation.
- 'reference_solution' must be a COMPLETE, CORRECT Python function definition (as a
  string), named exactly by 'function_name'. It may import and use numpy and math.
- 'test_args' is a list of argument lists; each inner list holds the positional
  arguments for one call to the function. Choose simple, valid inputs.
- Make it concrete and computable. Do NOT make it open-ended or symbolic.
- Write 'reference_solution' as valid, properly-indented Python using real newlines
  (\\n) between lines and 4-space indentation for the function body. Do NOT put the
  whole function on one line. Do NOT use semicolons to separate statements. It must
  be code that runs as-is with exec().
- Choose 'test_args' that are valid and safe for function evaluation: no zero
  denominators, no empty arrays, no out-of-range indices. Each test call must
  run without error.
- Make sure the number and type of arguments in each test_args entry exactly match
  the function's parameters.
- Prefer simple, small, clearly-valid inputs (e.g. positive numbers, small arrays)."""


def build_prompt_b(example: dict) -> str:
    return f"""Create ONE NEW heat-transfer coding problem in the same style as the example.

Rules:
- Change the scenario and numbers; do NOT copy the example.
{RULES_B}

Example:
  instruction: {example.get('instruction', '')}
  function_name: {example.get('function_name', '')}
  reference_solution: {example.get('reference_solution', '')}

Produce JSON with fields: instruction, function_name, reference_solution, test_args."""


def build_repair_prompt_b(draft: dict, feedback: list) -> str:
    return f"""You wrote the heat-transfer coding problem below for a dataset. Automatic
checks found the problems listed under "Issues found". Revise it so every issue is fixed.

How to revise:
- Fix the physics and the code. Do NOT just change the description to match
  wrong code.
- Keep the same topic. You may change the description, the code and the test inputs.

Rules:
{RULES_B}

Your problem:
  instruction: {draft.get('instruction', '')}
  function_name: {draft.get('function_name', '')}
  reference_solution: {draft.get('reference_solution', '')}
  test_args: {json.dumps(draft.get('test_args', []))}

Issues found:
{_feedback_lines(feedback)}

Produce the revised problem as JSON with fields: instruction, function_name,
reference_solution, test_args."""


def _finalize_b(raw: dict, example: dict, model: str) -> dict:
    """
    Turn the model's JSON into a Type B problem: run the reference solution on
    each test_args entry to compute expected values, and add type and
    provenance. Raises on failure (load error, runtime error, no test args).
    """
    obj = dict(raw)
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
        raise ValueError("no test_args provided")
    obj["test_cases"] = test_cases
    obj.pop("test_args", None)  # the inputs now live in test_cases[i]["args"]
    obj["type"] = "B"
    _add_provenance(obj, example, model)
    return obj


def generate_one_b(example: dict, model: str) -> dict | None:
    """
    Ask the model for one Type B problem modeled on `example`. Expected values
    are computed by running the model's reference_solution on its test_args.
    Returns None if the call, the parse, loading or any test call fails.
    """
    try:
        raw = _chat_json(model, build_prompt_b(example), TYPE_B_SCHEMA, TYPE_B_TEMPERATURE)
        return _finalize_b(raw, example, model)
    except Exception as e:
        print(f"  type-B generation/parse failed: {e!r}")
        return None


GENERATORS = {"A": generate_one_a, "B": generate_one_b}

# Per type: (schema, generation prompt, repair prompt, finalize, generation temperature)
_TYPE_SPECS = {
    "A": (TYPE_A_SCHEMA, build_prompt_a, build_repair_prompt_a, _finalize_a, TYPE_A_TEMPERATURE),
    "B": (TYPE_B_SCHEMA, build_prompt_b, build_repair_prompt_b, _finalize_b, TYPE_B_TEMPERATURE),
}


###############################################################################
# Repair loop
###############################################################################

RESOLVE_FEEDBACK = ("An independent solver, given only the problem text, got a different "
                    "answer. Re-check the physics, the formula and every number against "
                    "the problem text.")


def quality_feedback(report: q.QualityReport) -> list:
    """
    Turn a failed QualityReport into feedback lines for the generator.
    The independent re-solve is reported without the solver's answer, so the
    generator cannot simply copy it and the re-solve stays independent.
    """
    lines = []
    for r in report.results:
        if r.passed:
            continue
        if r.name == "resolve_agreement":
            lines.append(RESOLVE_FEEDBACK)
        else:
            lines.append(f"{r.name}: {r.detail}")
    return lines


def generate_with_repair(
    seed: dict,
    model: str,
    max_repairs: int = 0,
    quality_checks: bool = True,
    judge_model: str | None = None,
) -> tuple[str, dict | None, list]:
    """
    Generate one problem from `seed`, check it, and if it fails, send the reasons
    back to the generator for up to `max_repairs` revisions.

    Each round's draft is checked from scratch: evaluate/run it, verify it, then
    (unless quality_checks=False) run the quality checks and, if judge_model is
    given, the independent re-solve.

    Returns (status, problem, history):
      status   'kept', 'unparsed', 'failed_verification' or 'failed_quality':
               for a problem that was not kept, the status of the last round in
               which the model returned a draft (a model call that fails during a
               repair round ends the loop and keeps the previous result);
      problem  the final problem dict (None if no round produced one);
      history  one entry per round: {"round", "status", "feedback"}.
    """
    schema, prompt_new, prompt_repair, finalize, temperature = _TYPE_SPECS[seed["type"]]
    history: list = []
    draft: dict | None = None
    feedback: list = []
    problem: dict | None = None
    status = "unparsed"

    for round_no in range(max_repairs + 1):
        # 1. Ask for a new draft (round 0) or a revision of the last one.
        try:
            if round_no == 0:
                draft = _chat_json(model, prompt_new(seed), schema, temperature)
            else:
                draft = _chat_json(model, prompt_repair(draft, feedback), schema,
                                   REPAIR_TEMPERATURE)
        except Exception as e:  # model call or JSON parse failed: nothing to revise
            history.append({"round": round_no, "status": "unparsed",
                            "feedback": [f"model call failed: {e!r}"]})
            break

        # 2. Compute the answer / expected values from the draft.
        try:
            problem = finalize(draft, seed, model)
        except Exception as e:
            status, feedback = "unparsed", [f"the {'check_expression' if seed['type'] == 'A' else 'code'} "
                                            f"failed to run: {e!r}"]
            history.append({"round": round_no, "status": status, "feedback": feedback})
            problem = None
            continue

        # 3. Deterministic verification.
        result = v.verify_problem(problem)
        if not result.passed:
            status, feedback = "failed_verification", [f"verification: {result.detail}"]
            history.append({"round": round_no, "status": status, "feedback": feedback})
            continue
        problem["verification"] = "passed"

        # 4. Semantic quality checks (and the re-solve, if a judge is given).
        if quality_checks:
            report = q.assess_problem(problem, judge_model=judge_model)
            if not report.passed:
                status, feedback = "failed_quality", quality_feedback(report)
                problem["quality"] = report.summary()
                history.append({"round": round_no, "status": status, "feedback": feedback})
                continue
            problem["quality_checks"] = "passed"
            if judge_model:
                problem["judge_model"] = judge_model

        status = "kept"
        history.append({"round": round_no, "status": status, "feedback": []})
        break

    if problem is not None and len(history) > 1:
        problem["repair_rounds"] = len(history) - 1
        problem["repair_history"] = history
    return status, problem, history


###############################################################################
# Generation
###############################################################################

def generate_dataset(
    seeds_path: Path,
    model: str,
    per_seed: int,
    quality_checks: bool = True,
    judge_model: str | None = None,
    max_repairs: int = 0,
) -> tuple[list, dict, list]:
    """
    Generate `per_seed` problems per seed and keep only those that pass
    deterministic verification and (unless quality_checks=False) the semantic
    quality checks. If `judge_model` is given, problems that pass the
    deterministic quality checks are also re-solved independently by that model.
    With max_repairs > 0, failing drafts are sent back to the generator with the
    failure reasons and revised up to max_repairs times.

    Returns (kept, stats, rejected). `rejected` holds problems that passed
    verification but failed a quality check (after any repairs), each with a
    'quality' summary and, if repaired, a 'repair_history'.
    """
    seeds = json.loads(Path(seeds_path).read_text(encoding="utf-8"))["seeds"]

    kept, rejected = [], []
    attempted = passed = failed = unparsed = failed_quality = 0
    kept_after_repair = repair_rounds = 0
    failed_by_check: Counter = Counter()

    for seed in seeds:
        stype = seed.get("type")
        if stype not in _TYPE_SPECS:
            print(f"  skipping seed {seed.get('id')}: unknown type {stype!r}")
            continue

        for _ in range(per_seed):
            attempted += 1
            status, problem, history = generate_with_repair(
                seed, model, max_repairs=max_repairs,
                quality_checks=quality_checks, judge_model=judge_model,
            )
            repair_rounds += len(history) - 1
            pid = f"gen-{seed['id']}-{attempted}"
            if problem is not None:
                problem["id"] = pid
            last = history[-1]["feedback"] if history else []

            if status == "kept":
                kept.append(problem)
                passed += 1
                if len(history) > 1:
                    kept_after_repair += 1
                note = f" after {len(history) - 1} repair(s)" if len(history) > 1 else ""
                print(f"  PASS {pid} ({stype}){note}")
            elif status == "failed_quality":
                failed_quality += 1
                failed_by_check.update(problem["quality"]["failed_checks"])
                rejected.append(problem)
                print(f"  FAIL quality {pid} ({stype}): {', '.join(problem['quality']['failed_checks'])}")
            elif status == "failed_verification":
                failed += 1
                print(f"  FAIL verification {pid} ({stype}): {'; '.join(last)}")
            else:
                unparsed += 1
                print(f"  FAIL generation {pid} ({stype}): {'; '.join(last)}")

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
        "max_repairs": max_repairs,
        "repair_rounds": repair_rounds,
        "kept_after_repair": kept_after_repair,
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
    ap.add_argument("--max-repairs", type=int, default=0,
                    help="How many times a failing draft is sent back to the generator "
                         "with the failure reasons (default 0: no repair).")
    ap.add_argument("--rejected-out", type=Path, default=REJECTED_PATH,
                    help="Where to write problems that failed a quality check.")
    args = ap.parse_args()
    if args.max_repairs < 0:
        ap.error("--max-repairs must be 0 or more")
    if args.judge_model and args.judge_model == args.model:
        print("WARNING: --judge-model equals --model; the re-solve is not independent.")

    print(f"Model: {args.model} | judge: {args.judge_model} | "
          f"max repairs: {args.max_repairs} | seeds: {args.seeds}")
    kept, stats, rejected = generate_dataset(
        args.seeds, args.model, args.per_seed,
        quality_checks=not args.no_quality_checks, judge_model=args.judge_model,
        max_repairs=args.max_repairs,
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
