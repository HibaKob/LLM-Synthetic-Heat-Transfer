"""
Semantic quality checks for generated heat-transfer problems.

`verification.py` proves that a problem is self-consistent: a Type A answer equals
its own check_expression, and a Type B reference runs on its own inputs. That is
necessary but not sufficient. A manual audit of the first full run (Nov 2026)
found that only about 7 of 80 verified problems were physically correct. The
failures were sign errors, operator-precedence bugs, wrong areas, values the
model invented, invented formulas, and unstable or unbounded Type B solvers.

This module adds a second, semantic filter in two layers:

  1. Deterministic checks (no model, cheap, run on every problem)
       Type A: answer finite and non-zero; magnitude quantities positive;
               every number in the formula traceable to the problem text;
               no precedence bugs between two literal numbers; answer units
               match the quantity asked for; answer in a plausible range;
               circular areas built from a stated diameter.
       Type B: outputs finite; every argument used; temperature outputs stay
               within the bounds set by the temperature inputs (maximum
               principle, when there is no heat source); explicit schemes
               respect their stability limits.

  2. Independent re-solve (needs a second model, run after layer 1 passes)
       A different model sees only the problem statement and writes its own
       check_expression (Type A) or function (Type B). The problem is kept only
       if both agree numerically. The comparison is still done by code, using
       the public verification API, so "model constructs, code computes" holds.

Each check returns a CheckResult. A problem passes when every check that ran
passed. Checks that do not apply to a problem (e.g. the bounds check on a
function that does not return temperatures) are recorded as skipped.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Callable, Optional

from syntheticheatransfer import verification as v


###############################################################################
# Result types
###############################################################################

@dataclass
class CheckResult:
    """Outcome of one check on one problem."""
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class QualityReport:
    """All check results for one problem."""
    problem_id: Optional[str]
    results: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list:
        return [r.name for r in self.results if not r.passed]

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> dict:
        """JSON-friendly summary, stored on rejected problems."""
        return {
            "passed": self.passed,
            "failed_checks": self.failed_checks,
            "details": {r.name: r.detail for r in self.results if not r.passed},
        }


def _ok(name: str, detail: str = "") -> CheckResult:
    return CheckResult(name, True, detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, False, detail)


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(name, True, detail, skipped=True)


###############################################################################
# Constants and vocabulary
###############################################################################

# Numbers a formula may use without them appearing in the problem text.
# Kept deliberately small: every entry is a physical constant, a unit factor,
# a geometric/exponent integer, or a coefficient from a standard correlation.
PHYSICAL_CONSTANTS = [
    9.81, 9.8,                                   # g
    5.67e-8, 5.669e-8, 5.6703e-8, 5.670374419e-8,  # Stefan-Boltzmann
    273.15, 273,                                 # C -> K
    287, 101325,                                 # air gas constant, 1 atm
    math.pi, math.e,
]
UNIT_FACTORS = [60, 3600, 100, 1000, 1e-2, 1e-3, 1e-4, 1e-6, 1e3, 1e5, 1e6]
SMALL_INTEGERS = [0, 1, 2, 3, 4, 6, 8, 9, 16, 27]   # exponents, D/4, D/6, Churchill-Chu
CORRELATION_COEFFICIENTS = [
    0.664, 0.332, 0.037, 0.023, 0.8, 0.4, 0.3, 0.5,   # flat plate, Dittus-Boelter
    0.387, 0.559, 0.6, 0.825, 0.492,                  # Churchill-Chu
    0.54, 0.15, 0.27, 0.52, 0.59, 0.1,                # free convection on plates
    3.66, 4.36, 871,                                  # internal flow, mixed BL
]
ALLOWED_CONSTANTS = PHYSICAL_CONSTANTS + UNIT_FACTORS + SMALL_INTEGERS + CORRELATION_COEFFICIENTS

# Factors used to relate a stated number to the one in the formula
# (cm/mm -> m, kJ -> J, cm^2/mm^2 -> m^2, bar -> Pa).
SCALE_FACTORS = [1e-2, 1e-3, 1e-4, 1e-6, 1e2, 1e3, 1e5, 1e6]

TEMPERATURE_UNITS = {"c", "k", "degc", "degreesc", "f", "celsius", "kelvin"}

# Unit families, keyed by a normalized unit string (see _normalize_units).
UNIT_FAMILIES = {
    "power": {"w", "watt", "watts", "kw"},
    "power_per_length": {"w/m"},
    "flux": {"w/m^2", "w/m2", "kw/m^2"},
    "h": {"w/m^2*k", "w/m^2k", "w/m2k", "w/m^2*c", "w/m^2c", "w/(m^2*k)", "w/m^2.k"},
    "dimensionless": {"dimensionless", "", "-", "none", "unitless"},
    "time": {"s", "sec", "secs", "second", "seconds", "min", "mins", "minute", "minutes",
             "h", "hr", "hrs", "hour", "hours"},
    "temperature": TEMPERATURE_UNITS,
    "resistance": {"k/w", "c/w", "m^2*k/w", "m^2k/w"},
    "diffusivity": {"m^2/s", "m2/s"},
    "per_length": {"1/m", "m^-1"},
    "mass_flow": {"kg/s", "g/s"},
    "energy": {"j", "kj", "mj"},
}

# What the question asks for -> allowed unit families. Order matters: the first
# matching pattern wins, so specific phrases come before general ones.
QUESTION_PATTERNS = [
    (r"\b(biot|nusselt|reynolds|grashof|rayleigh|prandtl|fourier) number\b", {"dimensionless"}),
    (r"\bview factor\b|\bf_?\d\d\b", {"dimensionless"}),
    (r"\b(fin )?(efficiency|effectiveness)\b", {"dimensionless"}),
    (r"\btime constant\b|\bhow long\b", {"time"}),
    (r"\bthermal diffusivity\b|\bdiffusivity\b", {"diffusivity"}),
    (r"\bfin parameter\b", {"per_length"}),
    (r"\bmass flow rate\b", {"mass_flow"}),
    (r"\bthermal resistance\b", {"resistance"}),
    (r"\b(convection|convective|heat transfer) coefficient\b|\bcoefficient h\b", {"h"}),
    (r"\bheat flux\b|\bradiosity\b|\bemissive power\b|\bper unit area\b|\bfor a unit area\b", {"flux"}),
    (r"\blmtd\b|\blog mean temperature difference\b", {"temperature"}),
    (r"\bper (unit|meter|metre) length\b|\bper meter\b", {"power_per_length"}),
    (r"\b(how much )?energy\b", {"energy"}),
    (r"\bheat (transfer )?rate\b|\brate of heat\b|\bheat loss\b|\bheat dissipated\b"
     r"|\bheat transfer\b|\bheat gain\b", {"power", "power_per_length"}),
    (r"\btemperature\b", {"temperature"}),   # last: many questions mention temperatures
]

# Plausible ranges per unit family, in the family's base unit.
PLAUSIBLE_RANGES = {
    "h": (0.1, 2e5),                # W/m^2K, from free convection in gas to boiling
    "temperature_c": (-273.15, 3000.0),
    "temperature_k": (0.0, 3300.0),
}
BOUNDED_DIMENSIONLESS = r"\bview factor\b|\bf_?\d\d\b|\befficiency\b|\beffectiveness\b"


###############################################################################
# Small helpers
###############################################################################

def _normalize_units(units: str) -> str:
    u = (units or "").strip().lower()
    for a, b in (("²", "^2"), ("³", "^3"), ("°", ""), ("·", "*"), (" ", ""),
                 ("deg", "deg"), ("m2", "m^2"), ("m^2^2", "m^2")):
        u = u.replace(a, b)
    return u


def _unit_family(units: str) -> Optional[str]:
    u = _normalize_units(units)
    for fam, members in UNIT_FAMILIES.items():
        if u in {_normalize_units(m) for m in members}:
            return fam
    return None


ASKING_VERB = r"\b(find|calculate|compute|determine|what|how|estimate|obtain)\b"


def _question_text(instruction: str) -> str:
    """
    The clause that asks the question: in each sentence containing an asking
    verb (find, calculate, ...), the text from that verb onwards. This skips
    givens in the same sentence ('If h is 20 W/m^2K, find the heat rate').
    Returns '' when nothing is asked.
    """
    clauses = []
    for sentence in re.split(r"(?<=[.?!])\s+", instruction.strip()):
        m = re.search(ASKING_VERB, sentence, re.I)
        if m:
            clauses.append(sentence[m.start():])
    return " ".join(clauses)


def _expected_families(instruction: str) -> Optional[set]:
    q = _question_text(instruction).lower()
    if not q:
        return None
    for pattern, families in QUESTION_PATTERNS:
        if re.search(pattern, q):
            return families
    return None


def _stated_numbers(text: str) -> list:
    """
    Absolute values of every number written in the problem text. Handles
    scientific notation written as '1.5*10^-5', '2.5 × 10^-7', '10^-4', '1e-5'.
    """
    t = text.replace("×", "*").replace("−", "-")
    nums = []

    def take_sci(m):
        mant = float(m.group(1)) if m.group(1) else 1.0
        nums.append(abs(mant * 10 ** int(m.group(2))))
        return " "

    t = re.sub(r"(\d+(?:\.\d+)?)\s*[*x]\s*10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?", take_sci, t)
    t = re.sub(r"()(?<![\d.])10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?", take_sci, t)
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", t):
        nums.append(abs(float(m.group(1))))
    return nums


def _is_number(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool))


def _is_power_of_ten(x: float) -> bool:
    return x > 0 and abs(math.log10(x) - round(math.log10(x))) < 1e-9 and x != 1


def _scaled_literal(node: ast.AST):
    """
    If node is '<...>*a*1e-k' (a number times a power of ten, possibly at the end
    of a longer product such as '0.8*5.67*1e-8'), return the (a, 1e-k) nodes.
    """
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
            and _is_number(node.right) and _is_power_of_ten(abs(float(node.right.value)))):
        return None
    left = node.left
    if _is_number(left):
        return left, node.right
    if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mult) and _is_number(left.right):
        return left.right, node.right
    return None


def _expression_numbers(expr: str) -> list:
    """
    The numeric literals of an expression, as a list of alternatives: each item
    is a list of readings, and each reading is a tuple of values. A plain literal
    has one reading. 'a*1e-k' has two: one number a*10^-k (the way the text
    writes '1.82*10^-5'), or the two numbers a and 1e-k (when 1e-k is itself a
    stated value, as in a volume of 10^-4 m^3).
    """
    tree = ast.parse(expr, mode="eval")
    folded_parts = set()
    items = []
    for node in ast.walk(tree):
        pair = _scaled_literal(node)
        if pair:
            mant, exp = pair
            a, b = abs(float(mant.value)), abs(float(exp.value))
            items.append([(a * b,), (a, b)])
            folded_parts |= {id(mant), id(exp)}
    for node in ast.walk(tree):
        if _is_number(node) and id(node) not in folded_parts:
            items.append([(abs(float(node.value)),)])
    return items


def _close(a: float, b: float, rel: float = 1e-3) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def _derivable_numbers(stated: list) -> list:
    """
    Numbers a formula may legitimately contain, given the stated ones:
      * each stated number after one conversion: unit scaling (mm -> m, kJ -> J),
        half or double (diameter <-> radius), C <-> K;
      * the sum or difference of two such numbers (a temperature difference,
        an outer radius from inner radius plus wall thickness);
      * the product of two numbers exactly as stated (an area from two sides).
    Division is deliberately not allowed: with it, almost any number can be
    'derived' (25/2.5 = 10 made an invented h = 10 look grounded).
    """
    pool = set(stated)
    pool |= {s * f for s in stated for f in SCALE_FACTORS}
    pool |= {s / 2 for s in stated} | {s * 2 for s in stated}
    kelvin = {s + 273.15 for s in stated} | {abs(s - 273.15) for s in stated}
    kelvin |= {s + 273 for s in stated}
    derived = set(pool) | kelvin
    for a, b in product(pool, repeat=2):   # C -> K values are not combined further
        derived.add(a + b)
        derived.add(abs(a - b))
    for a, b in product(stated, repeat=2):
        derived.add(a * b)
    return list(derived)


def _is_grounded(value: float, derivable: list) -> bool:
    if any(_close(value, c) for c in ALLOWED_CONSTANTS):
        return True
    return any(_close(value, d) for d in derivable)



###############################################################################
# Dimensional analysis
###############################################################################
# Dimensions are exponent vectors over (mass, length, time, temperature).
# Magnitude prefixes (mm, kJ) do not change dimensions, so they are ignored here;
# unit scaling is handled by the number-grounding check.

Dim = tuple  # (M, L, T, Theta) as Fractions
DIMENSIONLESS: Dim = (Fraction(0),) * 4

BASE_UNITS = {
    "kg": (1, 0, 0, 0), "g": (1, 0, 0, 0),
    "m": (0, 1, 0, 0), "cm": (0, 1, 0, 0), "mm": (0, 1, 0, 0), "km": (0, 1, 0, 0),
    "s": (0, 0, 1, 0), "sec": (0, 0, 1, 0), "seconds": (0, 0, 1, 0), "second": (0, 0, 1, 0),
    "min": (0, 0, 1, 0), "minutes": (0, 0, 1, 0), "minute": (0, 0, 1, 0),
    "h": (0, 0, 1, 0), "hr": (0, 0, 1, 0), "hours": (0, 0, 1, 0), "hour": (0, 0, 1, 0),
    "k": (0, 0, 0, 1), "c": (0, 0, 0, 1), "degc": (0, 0, 0, 1),
    "w": (1, 2, -3, 0), "kw": (1, 2, -3, 0), "watt": (1, 2, -3, 0), "watts": (1, 2, -3, 0),
    "j": (1, 2, -2, 0), "kj": (1, 2, -2, 0),
    "n": (1, 1, -2, 0),
    "pa": (1, -1, -2, 0), "kpa": (1, -1, -2, 0), "bar": (1, -1, -2, 0), "atm": (1, -1, -2, 0),
}
_UNIT_TOKEN = re.compile(r"(kg|kj|kw|kpa|km|mm|cm|min|sec|pa|bar|atm|degc|watts|watt|hr|[gmshkcwjn])"
                         r"(?:\^?(-?\d+))?")


def _dim(*e) -> Dim:
    return tuple(Fraction(x) for x in e)


def _dim_mul(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))


def _dim_div(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b))


def _dim_pow(a: Dim, p) -> Dim:
    return tuple(x * Fraction(p).limit_denominator(1000) for x in a)


def _parse_unit_part(part: str) -> Optional[Dim]:
    """Product of unit tokens with optional exponents, e.g. 'm^2*k', 'm2k', 'kg'."""
    part = part.replace("*", "").replace(".", "").replace("(", "").replace(")", "")
    if not part:
        return DIMENSIONLESS
    total = DIMENSIONLESS
    pos = 0
    while pos < len(part):
        m = _UNIT_TOKEN.match(part, pos)
        if not m:
            return None
        exp = int(m.group(2)) if m.group(2) else 1
        total = _dim_mul(total, _dim_pow(_dim(*BASE_UNITS[m.group(1)]), exp))
        pos = m.end()
    return total


def parse_units(units: str) -> Optional[Dim]:
    """
    Dimensions of a unit string such as 'W/m^2*K', 'W/m²K', 'J/kg*K', 'm^2*K/W',
    '1/m', 'dimensionless'. Engineering convention: everything after the first
    '/' is in the denominator ('W/m*K' means W/(m*K)). Returns None if a token
    is not recognized.
    """
    u = _normalize_units(units)
    if u in {"", "-", "dimensionless", "none", "unitless"}:
        return DIMENSIONLESS
    u = u.replace("m^2^2", "m^2")
    num, _, den = u.partition("/")
    if num == "1":
        num = ""
    a = _parse_unit_part(num)
    b = _parse_unit_part(den)
    if a is None or b is None:
        return None
    return _dim_div(a, b)


# Dimensions of the allowed constants (everything else in ALLOWED_CONSTANTS is a
# pure number: unit factors, integers, correlation coefficients, pi, e).
CONSTANT_DIMS = [
    ([5.67e-8, 5.669e-8, 5.6703e-8, 5.670374419e-8], _dim(1, 0, -3, -4)),  # W/m^2K^4
    ([9.81, 9.8], _dim(0, 1, -2, 0)),                                      # m/s^2
    ([273.15, 273], _dim(0, 0, 0, 1)),                                     # K
    ([287], _dim(0, 2, -2, -1)),                                           # J/kg*K
    ([101325], _dim(1, -1, -2, 0)),                                        # Pa
]

def _to_e_notation(text: str) -> str:
    """Rewrite '1.5*10^-5' and '10^-4' as '1.5e-5' and '1e-4'."""
    t = text.replace("×", "*").replace("−", "-")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[*x]\s*10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?",
               lambda m: f"{m.group(1)}e{int(m.group(2))}", t)
    t = re.sub(r"(?<![\d.])10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?",
               lambda m: f"1e{int(m.group(1))}", t)
    return t


_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"        # number
    r"(?:\s*[*x×]\s*10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?)?"     # optional *10^k
    r"\s*(?:°\s*)?"
    r"([A-Za-z°][A-Za-z0-9°²³^*/.()]*)?")                   # optional unit


TEMPERATURE: Dim = _dim(0, 0, 0, 1)
LENGTH: Dim = _dim(0, 1, 0, 0)


def _text_dims(text: str) -> dict:
    """
    Candidate dimensions for numbers in the problem text, in two tiers:
      exact:   numbers as written, with the unit after them ('0.6 W/m*K',
               '2 cm', '25 C', '1.5*10^-5 m^2/s'). A number with no recognizable
               unit after it is dimensionless (emissivity, Pr, a Nusselt number).
      derived: unit-scaled values (2 cm -> 0.02), half/double (radius/diameter),
               C -> K, sums/differences of two lengths or two temperatures,
               products of two lengths (an area).
    Returns {"exact": {value: set(dims)}, "derived": {value: set(dims)}}.
    """
    t = _to_e_notation(text)
    exact: dict = {}
    derived: dict = {}

    def put(table, value, d):
        table.setdefault(round(abs(value), 12), set()).add(d)

    stated = []
    for m in _NUMBER_WITH_UNIT.finditer(t):
        value = float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)
        unit = (m.group(3) or "").rstrip(".,;)")
        d = parse_units(unit) if unit else None
        if d is None:
            word = unit.lower()
            if word in {"meter", "meters", "metre", "metres"}:
                d = LENGTH
            elif not unit or not word[:1].isalpha() or word in {"and", "or", "for", "with",
                                                               "in", "of", "to", "at"}:
                d = DIMENSIONLESS
            else:
                continue  # unknown unit ('A', 'nodes'): leave this number unmapped
        put(exact, value, d)
        stated.append((abs(value), d))
    for value, d in stated:
        for f in SCALE_FACTORS + [0.5, 2]:
            put(derived, value * f, d)
        if d == TEMPERATURE:
            put(derived, value + 273.15, d)
            put(derived, value + 273, d)
    for (a, da), (b, db) in product(stated, repeat=2):
        if da == db and da in (TEMPERATURE, LENGTH):
            put(derived, a + b, da)
            put(derived, abs(a - b), da)
        if da == LENGTH and db == LENGTH:
            put(derived, a * b, _dim_mul(da, db))
    # lengths given in cm/mm combine with lengths in m (inner radius + wall thickness)
    lengths = [v * f for v, d in stated if d == LENGTH for f in (1, 1e-2, 1e-3)]
    for a, b in product(lengths, repeat=2):
        put(derived, a + b, LENGTH)
        put(derived, abs(a - b), LENGTH)
    return {"exact": exact, "derived": derived}


class _UnknownDim(Exception):
    """Raised when a leaf's dimensions cannot be determined."""


class _DimMismatch(Exception):
    """Raised when terms with different dimensions are added or subtracted."""


_DIMLESS_FUNCS = {"log", "log10", "log2", "exp", "tanh", "sinh", "cosh", "sin", "cos",
                  "tan", "atan", "asin", "acos", "erf", "erfc"}


def _lookup(table: dict, value: float) -> set:
    found = set()
    for k, ds in table.items():
        if _close(k, abs(value), 1e-6):
            found |= ds
    return found


def _number_candidates(value: float, text_dims: dict, exact_only: bool = False) -> list:
    """
    Possible dimensions of a literal. Every applicable reading is returned and
    the assignment search in expression_dimensions picks a consistent one:
      * the units written next to this exact number in the text;
      * a physical constant's units (sigma, g, 273.15, ...), or dimensionless
        for an allowed pure number (2 in 2*pi, 4 in D/4, 1000 in kJ -> J);
      * if the number is not written in the text: the units of a derived value
        (2 cm -> 0.02 m, a temperature difference, a radius from a diameter).
    """
    # Ordered from most to least natural reading, so the first assignment tried
    # (and the one reported on failure) uses the units written in the text.
    found = sorted(_lookup(text_dims["exact"], value))
    for values, d in CONSTANT_DIMS:
        if any(_close(value, c, 1e-3) for c in values):
            found.append(d)
    if any(_close(value, c) for c in ALLOWED_CONSTANTS):
        found.append(DIMENSIONLESS)
    if not exact_only and not _lookup(text_dims["exact"], value):
        found.extend(sorted(_lookup(text_dims["derived"], value)))
    return list(dict.fromkeys(found))


def _leaf_candidates(tree: ast.AST, text_dims: dict) -> dict:
    """
    Map id(node) -> list of candidate dims for every numeric leaf. A scaled
    literal 'a*1e-k' becomes one leaf (keyed by the Mult node) when its folded
    value is in the text; otherwise its two numbers are separate leaves.
    """
    cands: dict = {}
    folded_parts = set()
    for node in ast.walk(tree):
        pair = _scaled_literal(node)
        if pair:
            mant, exp = pair
            # Only an exact match counts: the text wrote '1.82*10^-5 m^2/s'.
            found = _lookup(text_dims["exact"], float(mant.value) * float(exp.value))
            if found:
                cands[id(node)] = sorted(found)
                folded_parts |= {id(mant), id(exp)}
    for node in ast.walk(tree):
        if _is_number(node) and id(node) not in folded_parts and id(node) not in cands:
            found = _number_candidates(float(node.value), text_dims)
            if not found:
                raise _UnknownDim(f"no unit found for {node.value:g}")
            cands[id(node)] = found
    return cands


def _eval_dim(node: ast.AST, assign: dict) -> Dim:
    """Dimensions of an expression, given a dimension for every numeric leaf."""
    if isinstance(node, ast.Expression):
        return _eval_dim(node.body, assign)
    if id(node) in assign:
        if isinstance(node, ast.BinOp) and isinstance(node.left, ast.BinOp):
            # folded '<rest> * a * 1e-k': the assigned units belong to a*1e-k only
            return _dim_mul(_eval_dim(node.left.left, assign), assign[id(node)])
        return assign[id(node)]
    if isinstance(node, ast.UnaryOp):
        return _eval_dim(node.operand, assign)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _eval_dim(node.left, assign)
            try:
                p = eval(compile(ast.Expression(node.right), "<exp>", "eval"),
                         v.expression_namespace())
            except Exception:  # noqa: BLE001
                raise _UnknownDim("non-numeric exponent")
            return _dim_pow(base, p)  # exponents are pure numbers by construction
        a = _eval_dim(node.left, assign)
        b = _eval_dim(node.right, assign)
        if isinstance(node.op, ast.Mult):
            return _dim_mul(a, b)
        if isinstance(node.op, ast.Div):
            return _dim_div(a, b)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if a != b:
                raise _DimMismatch(f"adds/subtracts {format_dim(a)} and {format_dim(b)} "
                                   f"in '{ast.unparse(node)}'")
            return a
        raise _UnknownDim(f"operator {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if fname == "sqrt" and len(node.args) == 1:
            return _dim_pow(_eval_dim(node.args[0], assign), Fraction(1, 2))
        if fname == "pow" and len(node.args) == 2:
            return _eval_dim(ast.BinOp(node.args[0], ast.Pow(), node.args[1]), assign)
        if fname in _DIMLESS_FUNCS and len(node.args) == 1:
            arg = _eval_dim(node.args[0], assign)
            if arg != DIMENSIONLESS:
                raise _DimMismatch(f"{fname}() of a quantity with units {format_dim(arg)}")
            return DIMENSIONLESS
        raise _UnknownDim(f"function {fname}")
    if isinstance(node, (ast.Attribute, ast.Name)) and getattr(node, "attr", getattr(node, "id", "")) in {"pi", "e"}:
        return DIMENSIONLESS
    raise _UnknownDim(type(node).__name__)


MAX_DIM_ASSIGNMENTS = 4096


def expression_dimensions(expr: str, instruction: str, target: Optional[Dim]):
    """
    Try every assignment of candidate units to the numbers in `expr`.
    Returns ("ok", dim) if some assignment is consistent and matches `target`
    (or, with target None, the first consistent result); ("fail", reason) if no
    assignment is consistent with `target`; ("skip", reason) if undetermined.
    """
    tree = ast.parse(expr, mode="eval")
    text_dims = _text_dims(instruction)
    try:
        cands = _leaf_candidates(tree, text_dims)
    except _UnknownDim as e:
        return "skip", str(e)
    keys = list(cands)
    n_combos = math.prod(len(cands[k]) for k in keys)
    if n_combos > MAX_DIM_ASSIGNMENTS:
        return "skip", f"too many unit readings ({n_combos})"
    first_error = None
    results = set()
    for combo in product(*(cands[k] for k in keys)):
        assign = dict(zip(keys, combo))
        try:
            got = _eval_dim(tree, assign)
        except _DimMismatch as e:
            first_error = first_error or str(e)
            continue
        except _UnknownDim as e:
            return "skip", str(e)
        if target is None or got == target:
            return "ok", got
        results.add(got)
    if results:
        shown = " or ".join(sorted(format_dim(d) for d in results))
        return "fail", f"formula gives {shown}"
    return "fail", first_error or "no consistent reading of the units"


_DIM_NAMES = ("kg", "m", "s", "K")


def format_dim(d: Dim) -> str:
    """Human-readable dimensions, e.g. 'kg m^2 s^-3' (= W)."""
    parts = []
    for name, e in zip(_DIM_NAMES, d):
        if e == 0:
            continue
        parts.append(name if e == 1 else f"{name}^{e}")
    return " ".join(parts) or "dimensionless"

###############################################################################
# Type A deterministic checks
###############################################################################

def check_a_asks_question(problem: dict) -> CheckResult:
    """The instruction must ask for something (find, calculate, compute, ...)."""
    name = "a_asks_question"
    if _question_text(problem.get("instruction", "")):
        return _ok(name)
    return _fail(name, "no sentence asks for a quantity (find/calculate/compute/...)")


def check_a_finite_nonzero(problem: dict) -> CheckResult:
    name = "a_finite_nonzero"
    x = float(problem["answer_numeric"])
    if not math.isfinite(x):
        return _fail(name, f"answer is not finite: {x}")
    if x == 0.0:
        return _fail(name, "answer is exactly zero")
    return _ok(name)


def check_a_positive_magnitude(problem: dict) -> CheckResult:
    """
    Rates, fluxes, coefficients, resistances, times, masses, energies and
    dimensionless groups are magnitudes and must be positive. Temperatures may be
    negative, so temperature answers are skipped.
    """
    name = "a_positive_magnitude"
    fam = _unit_family(problem.get("answer_units", ""))
    if fam == "temperature":
        return _skip(name, "temperature answers may be negative")
    x = float(problem["answer_numeric"])
    if x < 0:
        return _fail(name, f"negative answer {x:.6g} {problem.get('answer_units', '')}")
    return _ok(name)


def check_a_numbers_grounded(problem: dict) -> CheckResult:
    """
    Every number in check_expression must come from the problem text, possibly
    after a unit conversion or one arithmetic step, or be an allowed constant.
    Catches values the model invented (an assumed k, h, or ambient temperature).
    """
    name = "a_numbers_grounded"
    stated = _stated_numbers(problem.get("instruction", ""))
    derivable = _derivable_numbers(stated)
    unexplained = set()
    for readings in _expression_numbers(problem["check_expression"]):
        if not any(all(_is_grounded(n, derivable) for n in reading) for reading in readings):
            unexplained.add(readings[0][0])
    unexplained = sorted(unexplained)
    if unexplained:
        shown = ", ".join(f"{n:g}" for n in unexplained)
        return _fail(name, f"numbers not traceable to the problem text: {shown}")
    return _ok(name)


_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _leaf_positions(node: ast.AST, side: str) -> Optional[ast.AST]:
    """
    Rightmost (side='right') or leftmost (side='left') operand reached through
    + - * / only. '**' binds tighter than everything, so '400**4 - 300**4'
    stops at the powers and is never flagged.
    """
    while isinstance(node, ast.BinOp) and isinstance(node.op, _ARITH):
        node = node.right if side == "right" else node.left
    return node


def _is_product(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div))


def _is_constant_sum(node: ast.AST) -> bool:
    """True for a constant, or constants joined only by + and -."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_constant_sum(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_constant_sum(node.left) and _is_constant_sum(node.right)
    return False


def check_a_precedence(problem: dict) -> CheckResult:
    """
    Flags 'a - b' or 'a + b' between two literal numbers, written without
    brackets, where one side is a bare number and the other is a product or
    quotient that grabbed the neighbouring number: '100*A*250-20' (meant
    100*A*(250-20)) or '80-20/(R1+R2)' (meant (80-20)/(R1+R2)). A sum of two
    products, such as 'x/20 + 1/8', is a normal sum of terms and is not flagged.
    """
    name = "a_precedence"
    expr = problem["check_expression"]
    tree = ast.parse(expr, mode="eval")
    lines = expr.splitlines() or [expr]
    flat = "\n".join(lines)
    offsets = [0]
    for ln in lines[:-1]:
        offsets.append(offsets[-1] + len(ln) + 1)

    def pos(node, end=False):
        line = (node.end_lineno if end else node.lineno) - 1
        col = node.end_col_offset if end else node.col_offset
        return offsets[line] + col

    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))):
            continue
        left_leaf = _leaf_positions(node.left, "right")
        right_leaf = _leaf_positions(node.right, "left")
        if not (isinstance(left_leaf, ast.Constant) and isinstance(right_leaf, ast.Constant)):
            continue
        between = flat[pos(left_leaf, end=True):pos(right_leaf)]
        if "(" in between or ")" in between:
            continue  # the author grouped it explicitly
        lone_left = _is_constant_sum(node.left) and _is_product(node.right)
        lone_right = _is_product(node.left) and _is_constant_sum(node.right)
        if not (lone_left or lone_right):
            continue  # both bare numbers, or both products: no precedence trap
        snippet = flat[pos(left_leaf):pos(right_leaf, end=True)]
        return _fail(name, f"'{snippet}' is split by operator precedence; brackets missing?")
    return _ok(name)


def check_a_units_match_question(problem: dict) -> CheckResult:
    """The answer's units must belong to the family of the quantity asked for."""
    name = "a_units_match_question"
    expected = _expected_families(problem.get("instruction", ""))
    if expected is None:
        return _skip(name, "could not tell which quantity is asked for")
    fam = _unit_family(problem.get("answer_units", ""))
    if fam is None:
        return _fail(name, f"unrecognized answer units '{problem.get('answer_units', '')}'")
    if fam not in expected:
        return _fail(name, f"question asks for {sorted(expected)} but units are "
                           f"'{problem.get('answer_units', '')}' ({fam})")
    return _ok(name)


def check_a_plausible_range(problem: dict) -> CheckResult:
    """Coarse physical ranges: h, temperatures, and bounded dimensionless groups."""
    name = "a_plausible_range"
    x = float(problem["answer_numeric"])
    units = _normalize_units(problem.get("answer_units", ""))
    fam = _unit_family(units)
    q = _question_text(problem.get("instruction", "")).lower()
    if fam == "h":
        lo, hi = PLAUSIBLE_RANGES["h"]
        if not lo <= x <= hi:
            return _fail(name, f"h = {x:.4g} W/m^2K is outside [{lo:g}, {hi:g}]")
        return _ok(name)
    if fam == "temperature":
        key = "temperature_k" if units in {"k", "kelvin"} else "temperature_c"
        lo, hi = PLAUSIBLE_RANGES[key]
        if not lo <= x <= hi:
            return _fail(name, f"temperature {x:.4g} {problem.get('answer_units')} "
                               f"is outside [{lo:g}, {hi:g}]")
        return _ok(name)
    if fam == "dimensionless" and re.search(BOUNDED_DIMENSIONLESS, q):
        if not 0.0 <= x <= 1.0:
            return _fail(name, f"{x:.4g} should lie in [0, 1]")
        return _ok(name)
    return _skip(name, "no range defined for this quantity")


def check_a_circular_area(problem: dict) -> CheckResult:
    """
    If the problem states a diameter D (and is not about a sphere), the formula
    must not square D next to pi without dividing by 4: pi*D**2 and pi*D*D are
    four times the cross-section. A sphere's surface area pi*D**2 is correct,
    so spheres are skipped.
    """
    name = "a_circular_area"
    text = problem.get("instruction", "")
    if re.search(r"\bspher", text, re.I):
        return _skip(name, "sphere: pi*D^2 is a valid surface area")
    diameters = []
    for m in re.finditer(r"diameter\s*(?:of|=|is)?\s*(?:D\s*=\s*)?(\d+(?:\.\d+)?)\s*(mm|cm|m)\b",
                         text, re.I):
        val = float(m.group(1)) * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[m.group(2).lower()]
        diameters.append(val)
    if not diameters:
        return _skip(name, "no diameter stated")
    expr = problem["check_expression"].replace(" ", "")
    if "pi" not in expr or re.search(r"/\(?4\b|/4\.0", expr):
        return _ok(name)
    num = r"\(?(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)?"
    for m in re.finditer(num + r"\*\*2|" + num + r"\*" + num, expr):
        if m.group(1) is not None:
            vals = [float(m.group(1))]
        else:
            if not _close(float(m.group(2)), float(m.group(3))):
                continue
            vals = [float(m.group(2))]
        if any(_close(vals[0], d) for d in diameters):
            return _fail(name, f"pi*D^2 with D = {vals[0]:g} m and no /4 "
                               f"(circle area is pi*D^2/4)")
    return _ok(name)


def check_a_dimensions(problem: dict) -> CheckResult:
    """
    Dimensional analysis of check_expression, using the unit written next to
    each number in the problem text. Fails when terms with different units are
    added or subtracted (e.g. '100*A*250 - 20', W minus K), when log/exp/tanh
    get an argument with units, or when the result does not have the dimensions
    of answer_units (e.g. k*A*dT without a thickness gives W*m, not W).
    Ambiguous numbers (the same value given with two units) are resolved by
    trying every reading; the check fails only if no reading works. Skipped when
    a number's unit cannot be identified.
    """
    name = "a_dimensions"
    target = parse_units(problem.get("answer_units", ""))
    if target is None:
        return _skip(name, f"unrecognized answer units '{problem.get('answer_units')}'")
    status, info = expression_dimensions(problem["check_expression"],
                                         problem.get("instruction", ""), target)
    if status == "skip":
        return _skip(name, f"dimensions not determined: {info}")
    if status == "fail":
        return _fail(name, f"{info}; answer units '{problem.get('answer_units')}' "
                           f"are {format_dim(target)}")
    return _ok(name)


TYPE_A_CHECKS: list = [
    check_a_asks_question,
    check_a_finite_nonzero,
    check_a_positive_magnitude,
    check_a_numbers_grounded,
    check_a_precedence,
    check_a_units_match_question,
    check_a_dimensions,
    check_a_plausible_range,
    check_a_circular_area,
]


###############################################################################
# Type B deterministic checks
###############################################################################

def _flatten(x) -> list:
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


def _function_def(problem: dict) -> Optional[ast.FunctionDef]:
    tree = ast.parse(problem["reference_solution"])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == problem["function_name"]:
            return node
    return None


def _param_names(problem: dict) -> list:
    fn = _function_def(problem)
    return [a.arg for a in fn.args.args] if fn else []


def check_b_outputs_finite(problem: dict) -> CheckResult:
    name = "b_outputs_finite"
    for case in problem["test_cases"]:
        for val in _flatten(case["expected"]):
            try:
                ok = math.isfinite(float(val))
            except (TypeError, ValueError):
                return _fail(name, f"{case['name']}: non-numeric output {val!r}")
            if not ok:
                return _fail(name, f"{case['name']}: non-finite output {val!r}")
    return _ok(name)


def check_b_all_args_used(problem: dict) -> CheckResult:
    """
    A function that ignores most of its inputs does not implement what it
    describes (e.g. 'h = k/L' with T_wall, T_ambient and h as unused arguments).
    One or two physically irrelevant inputs are legitimate, though: in steady
    1-D conduction with no generation, node temperatures do not depend on L or k
    (seed cond-B-01). So the check fails only when at least half the arguments
    are unused, and otherwise reports the unused ones without failing.
    """
    name = "b_all_args_used"
    fn = _function_def(problem)
    if fn is None:
        return _fail(name, f"function '{problem['function_name']}' not found")
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    params = [a.arg for a in fn.args.args]
    unused = [p for p in params if p not in used]
    if params and len(unused) * 2 >= len(params):
        return _fail(name, f"{len(unused)} of {len(params)} arguments unused: {', '.join(unused)}")
    return _ok(name, f"unused arguments: {', '.join(unused)}" if unused else "")


TEMPERATURE_OUTPUT = (r"\btemperatures\b|\btemperature (profile|history|distribution|at)\b"
                      r"|\bfinal temperature\b")
HEAT_SOURCE = r"generat|\bsource\b|q_gen|heat flux|\bflux\b|heater|\bpower\b"
TEMPERATURE_ARG = re.compile(r"^T(_|\d|$)")


def check_b_temperature_bounds(problem: dict) -> CheckResult:
    """
    Maximum principle: with no heat source, steady and transient conduction and
    convective cooling never produce temperatures outside the range of the
    initial, boundary and ambient temperatures. Applies when the function returns
    temperatures and its arguments include temperatures (names starting with T).
    """
    name = "b_temperature_bounds"
    text = problem.get("instruction", "")
    if not re.search(TEMPERATURE_OUTPUT, text, re.I):
        return _skip(name, "function does not return temperatures")
    if re.search(HEAT_SOURCE, text, re.I):
        return _skip(name, "problem has a heat source; bounds do not apply")
    params = _param_names(problem)
    t_idx = [i for i, p in enumerate(params) if TEMPERATURE_ARG.match(p)]
    if not t_idx:
        return _skip(name, "no temperature arguments")
    for case in problem["test_cases"]:
        temps = [float(case["args"][i]) for i in t_idx if i < len(case["args"])]
        lo, hi = min(temps), max(temps)
        tol = 1e-6 * max(hi - lo, 1.0)
        out = [float(x) for x in _flatten(case["expected"])]
        bad = [x for x in out if x < lo - tol or x > hi + tol]
        if bad:
            worst = max(bad, key=lambda x: max(lo - x, x - hi))
            return _fail(name, f"{case['name']}: output {worst:.4g} outside input "
                               f"temperature range [{lo:g}, {hi:g}]")
    return _ok(name)


def check_b_explicit_stability(problem: dict) -> CheckResult:
    """
    Stability limits of explicit schemes, evaluated on each test case:
      FTCS:          alpha*dt/dx^2 <= 0.5, with dx = L/(n-1)
      forward Euler: dt/tau <= 1 (no overshoot past the ambient value)
    Runs only when the parameter names make the quantities identifiable.
    """
    name = "b_explicit_stability"
    params = _param_names(problem)
    idx = {p: i for i, p in enumerate(params)}
    length = next((p for p in ("L", "W", "length") if p in idx), None)
    checked = False
    for case in problem["test_cases"]:
        a = case["args"]
        if {"alpha", "dt", "n"} <= idx.keys() and length:
            checked = True
            n = float(a[idx["n"]])
            dx = float(a[idx[length]]) / (n - 1) if n > 1 else float("inf")
            r = float(a[idx["alpha"]]) * float(a[idx["dt"]]) / dx ** 2
            if r > 0.5 + 1e-12:
                return _fail(name, f"{case['name']}: FTCS r = alpha*dt/dx^2 = {r:.3g} > 0.5")
        if {"tau", "dt"} <= idx.keys():
            checked = True
            ratio = float(a[idx["dt"]]) / float(a[idx["tau"]])
            if ratio > 1 + 1e-12:
                return _fail(name, f"{case['name']}: dt/tau = {ratio:.3g} > 1")
    return _ok(name) if checked else _skip(name, "no explicit-scheme parameters found")


TYPE_B_CHECKS: list = [
    check_b_outputs_finite,
    check_b_all_args_used,
    check_b_temperature_bounds,
    check_b_explicit_stability,
]


###############################################################################
# Deterministic entry point
###############################################################################

def _run(check: Callable, problem: dict) -> CheckResult:
    """Run one check; a crash inside a check is reported as a failure, not raised."""
    try:
        return check(problem)
    except Exception as e:  # noqa: BLE001 - one bad problem must not stop a batch
        return _fail(check.__name__.removeprefix("check_"), f"check raised {e!r}")


def check_problem(problem: dict) -> QualityReport:
    """Run every deterministic check for the problem's type."""
    checks = {"A": TYPE_A_CHECKS, "B": TYPE_B_CHECKS}.get(problem.get("type"), [])
    report = QualityReport(problem.get("id"))
    if not checks:
        report.results.append(_fail("type", f"unknown type {problem.get('type')!r}"))
    for check in checks:
        report.results.append(_run(check, problem))
    return report


###############################################################################
# Independent re-solve (second model)
###############################################################################

RESOLVE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "check_expression": {"type": "string"},
    },
    "required": ["reasoning", "check_expression"],
}

RESOLVE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_solution": {"type": "string"},
    },
    "required": ["reference_solution"],
}


def build_resolve_prompt_a(problem: dict) -> str:
    return f"""Solve this heat-transfer problem. Do not compute the final number yourself.
Instead, write a single Python expression that computes it.

Rules:
- Use only the numbers given in the problem, and standard physical constants.
- If the problem does not give enough information to solve it, write the
  expression 'None'.
- Use math.* or np.* explicitly (math.pi, math.log, math.sqrt, math.exp).
- The answer must be in {problem.get('answer_units', 'the requested units')}.

Problem:
{problem['instruction']}

Produce JSON with fields: reasoning, check_expression."""


def build_resolve_prompt_b(problem: dict) -> str:
    return f"""Write the Python function described below.

Rules:
- Name the function exactly '{problem['function_name']}' and use exactly the
  parameters named in the description, in the same order.
- You may import math and numpy. Use real newlines and 4-space indentation.
- Return exactly what the description asks for.

Description:
{problem['instruction']}

Produce JSON with field: reference_solution."""


def _default_chat():
    import ollama  # imported lazily so the deterministic checks need no model
    return ollama.chat


def _ask(chat: Callable, model: str, prompt: str, schema: dict) -> dict:
    resp = chat(model=model, messages=[{"role": "user", "content": prompt}],
                format=schema, options={"temperature": 0.0})
    return json.loads(resp["message"]["content"])


def resolve_type_a(problem: dict, model: str, chat: Optional[Callable] = None) -> CheckResult:
    """
    A second model writes its own check_expression from the problem text alone.
    Passes if it agrees with the generator's answer within the problem's
    tolerance. The comparison uses verification.verify_type_a.
    """
    name = "resolve_agreement"
    try:
        obj = _ask(chat or _default_chat(), model, build_resolve_prompt_a(problem), RESOLVE_A_SCHEMA)
        expr = (obj.get("check_expression") or "").strip()
    except Exception as e:  # noqa: BLE001
        return _fail(name, f"re-solve call failed: {e!r}")
    if expr in {"", "None"}:
        return _fail(name, "second model judged the problem unsolvable as stated")
    trial = copy.deepcopy(problem)
    trial["check_expression"] = expr
    result = v.verify_type_a(trial)
    if result.passed:
        return _ok(name, f"independent expression agrees: {expr}")
    return _fail(name, f"independent expression '{expr}' disagrees: {result.detail}")


def resolve_type_b(problem: dict, model: str, chat: Optional[Callable] = None) -> CheckResult:
    """
    A second model writes its own function from the description alone. Passes if
    it reproduces the generator's outputs on every test case (verify_type_b with
    the second model's code as the candidate).
    """
    name = "resolve_agreement"
    try:
        obj = _ask(chat or _default_chat(), model, build_resolve_prompt_b(problem), RESOLVE_B_SCHEMA)
        code = obj.get("reference_solution") or ""
    except Exception as e:  # noqa: BLE001
        return _fail(name, f"re-solve call failed: {e!r}")
    result = v.verify_type_b(problem, candidate_solution=code)
    if result.passed:
        return _ok(name, "independent implementation agrees on all test cases")
    return _fail(name, f"independent implementation disagrees: {result.detail}")


def resolve_problem(problem: dict, model: str, chat: Optional[Callable] = None) -> CheckResult:
    if problem.get("type") == "A":
        return resolve_type_a(problem, model, chat)
    if problem.get("type") == "B":
        return resolve_type_b(problem, model, chat)
    return _fail("resolve_agreement", f"unknown type {problem.get('type')!r}")


###############################################################################
# Full pipeline entry point
###############################################################################

def assess_problem(problem: dict, judge_model: Optional[str] = None,
                   chat: Optional[Callable] = None) -> QualityReport:
    """
    Deterministic checks first; if they all pass and a judge model is given,
    run the independent re-solve. The re-solve is skipped for problems that
    already failed, to save model calls.
    """
    report = check_problem(problem)
    if judge_model and report.passed:
        report.results.append(resolve_problem(problem, judge_model, chat))
    return report
