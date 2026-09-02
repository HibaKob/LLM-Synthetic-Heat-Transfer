"""
Synthetic problem generation using Ollama with structured output and
deterministic verification.
 
Design principles considering Ollam and other local basic llm restrictions:
  * The model constructs problems: scenario + formula/code
  * The model is not required to do any arithmetic to supply 
    the final numeric answer (Type A) or the test expected 
    values (Type B). The code computes those:
      - Type A: answer_numeric = eval(check_expression)
      - Type B: expected = reference_solution(*test_args)
    This makes every kept problem self-consistent by construction.
  * Ollama's `format` (JSON schema) constrains output so it parses reliably.
  * Every generated problem is run through the deterministic verifier; only
    problems that pass are kept.
 
Verification semantics:
  * Type A "passed" = check_expression is valid/evaluable and the answer matches it.
  * Type B "passed" = reference_solution runs deterministically on its test args.
    (Neither guarantees the formula/code truly matches the wording - that is the
    job of a future LLM-as-judge stage.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import ollama 
from syntheticheatransfer import verification as v


###############################################################################
# The schema the model will follow
###############################################################################

    
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
  'math.sqrt(...)', 'math.log(...)', 'np.pi', 'np.linalg.inv(...), np.array(...). Do NOT use 
  bare 'pi', the symbol, or any ame not from the math or numpy modules. Use only numbers, math.*,
  and np.* functions. 
- 'check_expression' must contain the ACTUAL numbers from your problem and must
  compute the numeric answer. The answer will be computed from this expression.
 
Example problem:
  instruction: {example.get('instruction', '')}
  check_expression: {example.get('check_expression', '')}
  answer_units: {example.get('answer_units', '')}
 
Now produce a new problem as JSON with fields: instruction, reasoning,
answer_units, check_expression."""
 
 
def generate_one_a(example: dict, model: str) -> dict | None:
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt_a(example)}],
            format=TYPE_A_SCHEMA,
            options={"temperature": 0.6},
        )
        obj = json.loads(resp["message"]["content"])
        obj["answer_numeric"] = float(eval(obj["check_expression"], v.expression_namespace()))
    except Exception as e:
        print(f"  type-A generation/parse failed: {e!r}")
        return None
 
    obj["type"] = "A"
    obj["provenance"] = "llm_generated_verified"
    obj["source_seed_id"] = example.get("id")
    obj["answer_tolerance_rel"] = 0.02
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
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": build_prompt_b(example)}],
            format=TYPE_B_SCHEMA,
            options={"temperature": 0.3},
        )
        obj = json.loads(resp["message"]["content"])
        fn = v.load_solution(obj["reference_solution"], obj["function_name"])
        test_cases = []
        for i, args in enumerate(obj["test_args"]):
            expected = fn(*list(args))
            test_cases.append({
                "name": f"case_{i + 1}",
                "args": list(args),
                "expected": expected,
                "tol_rel": 1e-6,
            })
        if not test_cases:
            print("  type-B: no test_args provided; dropping.")
            return None
        obj["test_cases"] = test_cases
    except Exception as e:
        print(f"  type-B generation/parse failed: {e!r}")
        return None
 
    obj["type"] = "B"
    obj["provenance"] = "llm_generated_verified"
    obj["source_seed_id"] = example.get("id")
    return obj


###############################################################################
# Generation
###############################################################################


def generate_dataset(seeds_path: Path, model: str, per_seed: int) -> tuple[list, dict]:
    seeds = json.loads(Path(seeds_path).read_text())["seeds"]
 
    kept = []
    attempted = passed = failed = unparsed = 0
 
    for seed in seeds:
        stype = seed.get("type")
        for _ in range(per_seed):
            attempted += 1
            if stype == "A":
                problem = generate_one_a(seed, model)
            elif stype == "B":
                problem = generate_one_b(seed, model)
            else:
                continue
 
            if problem is None:
                unparsed += 1
                continue
 
            problem["id"] = f"gen-{seed['id']}-{attempted}"
            result = v.verify_problem(problem)
            if result.passed:
                problem["verification"] = "passed"
                kept.append(problem)
                passed += 1
                print(f"  PASS {problem['id']} ({stype})")
            else:
                failed += 1
                print(f"  FAIL {problem['id']} ({stype}): {result.detail}")
 
    stats = {
        "attempted": attempted,
        "passed": passed,
        "failed_verification": failed,
        "unparsed": unparsed,
    }
    return kept, stats
 

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data"

SEEDS_DIR = DATA_DIR / "seeds" / "heat_transfer_seeds.json"
OUTPUT_PATH = DATA_DIR / "generated" / "generated_examples.json"

def main() -> None:
    v.create_output_folder(REPO_ROOT, "data")           
    v.create_output_folder(DATA_DIR, "generated")

    ap = argparse.ArgumentParser(description="Direct-Ollama verifiable problem generation.")
    ap.add_argument("--seeds", type=Path, default=SEEDS_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--model", default="llama3.1:latest")
    ap.add_argument("--per-seed", type=int, default=2,
                    help="How many new problems to generate per seed.")
    args = ap.parse_args()
 
    print(f"Model: {args.model} | seeds: {args.seeds}")
    kept, stats = generate_dataset(args.seeds, args.model, args.per_seed)
    print(f"\nStats: {stats}")
    print("Kept by type:", dict(Counter(p["type"] for p in kept)))
 
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "domain": "heat_transfer",
        "description": "LLM-generated problems that passed deterministic verification.",
        "generation_stats": stats,
        "seeds": kept,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(kept)} verified examples to {args.out}")
 
 
if __name__ == "__main__":
    main()