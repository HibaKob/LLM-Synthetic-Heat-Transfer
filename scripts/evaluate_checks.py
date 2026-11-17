"""
Measure the semantic quality checks against hand-labelled problems.

Runs the deterministic checks in quality.py on
  * a generated file whose items were labelled by hand (default: the Nov 2026
    audit fixture), reporting how many wrong problems each check catches and
    whether any correct problem is rejected;
  * the human-authored seeds, which should all pass (every failure is either a
    seed to fix or a limit of the checks).

Usage:
    python scripts/evaluate_checks.py
    python scripts/evaluate_checks.py --generated data/generated/generated_examples.json \
        --labels my_labels.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from syntheticheatransfer import quality as q

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def evaluate_labelled(problems: list, labels: dict) -> None:
    caught_by = Counter()
    false_alarms = defaultdict(list)
    missed = []
    totals = Counter()
    for p in problems:
        verdict = labels[p["id"]]["verdict"]
        wrong = verdict == "wrong"
        totals[verdict] += 1
        report = q.check_problem(p)
        for name in report.failed_checks:
            if wrong:
                caught_by[name] += 1
            else:
                false_alarms[name].append(p["id"])
        if wrong and report.passed:
            missed.append(p)

    n_wrong = totals["wrong"]
    n_caught = n_wrong - len(missed)
    n_good = sum(n for v, n in totals.items() if v != "wrong")
    n_false = len({i for ids in false_alarms.values() for i in ids})
    print(f"Labelled problems: {len(problems)} ({dict(totals)})")
    print(f"Wrong problems caught: {n_caught} of {n_wrong} ({100 * n_caught / max(n_wrong, 1):.0f}%)")
    print(f"Correct/borderline problems rejected: {n_false} of {n_good}\n")

    print(f"{'check':<26}{'wrong caught':>14}{'false alarms':>14}")
    names = sorted(set(caught_by) | set(false_alarms), key=lambda n: -caught_by[n])
    for name in names:
        print(f"{name:<26}{caught_by[name]:>14}{len(false_alarms.get(name, [])):>14}")
    for name, ids in false_alarms.items():
        if ids:
            print(f"  false alarm ({name}): {', '.join(ids)}")

    by_category = Counter(c for p in missed for c in labels[p["id"]]["categories"])
    print(f"\nMissed wrong problems by audit category: {dict(by_category)}")
    print("(These need the independent re-solve: --judge-model in generate.py.)")


def evaluate_seeds(seeds: list) -> None:
    flagged = [(s["id"], q.check_problem(s)) for s in seeds]
    flagged = [(i, r) for i, r in flagged if not r.passed]
    print(f"\nSeeds: {len(seeds)}, flagged: {len(flagged)}")
    for sid, report in flagged:
        for name, detail in report.summary()["details"].items():
            print(f"  {sid}: {name}: {detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--generated", type=Path,
                    default=FIXTURES / "generated_examples_2026-11-02.json")
    ap.add_argument("--labels", type=Path, default=FIXTURES / "audit_labels.json")
    ap.add_argument("--seeds", type=Path,
                    default=REPO_ROOT / "data" / "seeds" / "heat_transfer_seeds.json")
    args = ap.parse_args()

    problems = json.loads(args.generated.read_text())["seeds"]
    labels = json.loads(args.labels.read_text())["labels"]
    problems = [p for p in problems if p["id"] in labels]
    evaluate_labelled(problems, labels)
    if args.seeds.exists():
        evaluate_seeds(json.loads(args.seeds.read_text())["seeds"])


if __name__ == "__main__":
    main()
