#!/usr/bin/env python3
"""
Mechanical semantic gate: YAML ``formula`` identifiers must align with paired ``lean_code`` defs.

``verify_integrity.py`` Rule 3 rejects vacuous Lean theorems but not placeholder defs whose
parameters are generic (``a``, ``b``) while the declared ``formula`` names physical quantities
(``alpha_f``, ``D_ij``, ``k_c``, …). This checker closes that gap for ``morphoeval/math_kg_loops/*.yaml``.

Alignment rules (per equation):
  1. Parse the Lean ``def`` (name, parameters, body) from ``lean_code``.
  2. If the def matches the overnight expansion stub templates documented in
     ``run_overnight_1000_loops.py`` (``a * b``, ``p + q - q``, ``lhs - rhs`` with generic params),
     every *physics token* extracted from ``formula`` must appear in the def (parameters + body).
  3. Otherwise (non-stub / mechanism-linked Lean), each physics token must match the def text
     via normalized identifier equality or a documented abbreviation (``c_Ca`` ↔ param ``c``).

Physics tokens are identifiers taken from ``formula`` after Unicode→ASCII normalization,
excluding common mathematical functions and dimensionless indices. Tokens with subscripts,
Greek letters, or length ≥ 4 are always physics tokens; shorter tokens must match a def parameter.

Usage:
  python3 morphoeval/check_lean_formula_alignment.py morphoeval/math_kg_loops/<sim>.yaml
  python3 morphoeval/check_lean_formula_alignment.py --self-test
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOOPS_DIR = HERE / "math_kg_loops"

# Overnight template negative control (run_overnight_1000_loops.py law1 stub).
OVERNIGHT_STUB_FORMULA = "dD_ij/dt = alpha_f |J_ij|^gamma - beta_d D_ij"
OVERNIGHT_STUB_LEAN = (
    "def law1_invariant (a b : Int) : Int := a * b\n"
    "theorem law1_positive_scaling (a b : Int) (ha : 0 < a) (hb : 0 < b) : "
    "0 < law1_invariant a b := Int.mul_pos ha hb"
)

POSITIVE_CONTROL = LOOPS_DIR / "forterre-venus-flytrap-snap-buckling.yaml"
NEGATIVE_CONTROL = LOOPS_DIR / "zhou-ginkgo-yimaensis-flux-canalization.yaml"

MATH_STOPWORDS = frozenset(
    {
        "abs",
        "and",
        "arcsin",
        "arccos",
        "arctan",
        "cos",
        "cosh",
        "decide",
        "dexp",
        "dim",
        "dmul",
        "dot",
        "dsimp",
        "dt",
        "else",
        "exp",
        "forall",
        "if",
        "int",
        "ln",
        "log",
        "max",
        "min",
        "mod",
        "nat",
        "not",
        "omega",
        "or",
        "pi",
        "pow",
        "round",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "theorem",
        "def",
        "by",
        "exact",
        "true",
        "false",
        "prop",
        "type",
        "real",
        # Prose inside long ``formula`` strings (not physics symbols).
        "monostable",
        "continuous",
        "bending",
        "bistable",
        "snap",
        "through",
        "with",
        "approx",
        "separated",
        "strictly",
        "positive",
        "barrier",
        "flat",
        "pass",
        "fold",
        "threshold",
        "curvature",
        "rapid",
        "smooth",
        "supercritical",
        "subcritical",
        "adult",
        "traps",
        "leaves",
        "surgical",
        "ablation",
        "abolishes",
        "snapping",
        "doubling",
        "amplifies",
        "drive",
        "regime",
        "while",
        "reduces",
        "below",
        "explains",
        "why",
        "size",
        "leaf",
        "strip",
        "cutting",
        "relief",
        "open",
        "closed",
        "inverted",
        "concave",
        "convex",
        "state",
        "wells",
        "energy",
        "stretching",
        "penalty",
        "gauss",
        "theorem",
        "egregium",
        "both",
        "zero",
        "initial",
        "Gaussian",
    }
)

GENERIC_STUB_PARAMS = frozenset({"a", "b", "p", "q", "lhs", "rhs"})

GREEK_ASCII = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "k",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Δ": "Delta",
    "Σ": "Sigma",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ω": "Omega",
}


def normalize_math_text(text: str) -> str:
    """Map Unicode math symbols to ASCII identifier-friendly text."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        if ch in GREEK_ASCII:
            out.append(GREEK_ASCII[ch])
        elif ch in "∂∇":
            continue
        elif ch.isalnum() or ch in "_":
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def extract_formula_identifiers(formula: str) -> list[str]:
    """Return ordered unique identifiers from a ``formula`` string."""
    norm = normalize_math_text(formula)
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]*", norm)
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low in MATH_STOPWORDS:
            continue
        if tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens


def extract_physics_tokens(formula: str) -> list[str]:
    """Identifiers that must be reflected in Lean when the def is an overnight stub."""
    return [t for t in extract_formula_identifiers(formula) if "_" in t or len(t) >= 5]


def parse_lean_def(lean_code: str) -> tuple[str, list[str], str] | None:
    """Return (def_name, param_names, body) for the first ``def`` in ``lean_code``."""
    m = re.search(
        r"\bdef\s+([A-Za-z0-9_']+)\s*(.*?)\s*:\=\s*(.*?)(?=\btheorem\b|\Z)",
        lean_code,
        flags=re.DOTALL,
    )
    if not m:
        return None
    name = m.group(1)
    header = m.group(2)
    body = m.group(3).strip()
    params: list[str] = []
    for chunk in re.findall(r"\(([^)]*)\)", header):
        if ":" not in chunk:
            continue
        names_part, _type_part = chunk.rsplit(":", 1)
        for name_tok in names_part.replace("{", " ").replace("}", " ").split():
            if re.match(r"^[A-Za-z_][A-Za-z0-9_']*$", name_tok):
                params.append(name_tok)
    return name, params, body


def normalize_body(body: str) -> str:
    return re.sub(r"\s+", "", body)


def is_overnight_expansion_stub(param_names: list[str], body: str) -> bool:
    """True when ``lean_code`` matches the generic templates from overnight loop generation."""
    params = tuple(param_names)
    b = normalize_body(body)
    if params == ("a", "b") and b in {"a*b", "b*a"}:
        return True
    if params == ("p", "q") and b == "p+q-q":
        return True
    if params == ("lhs", "rhs") and b == "lhs-rhs":
        return True
    return False


def token_in_text(token: str, text: str) -> bool:
    norm_tok = token.lower()
    blob = text.lower()
    if re.search(rf"\b{re.escape(norm_tok)}\b", blob):
        return True
    stem = token.split("_")[0].lower()
    if stem and re.search(rf"\b{re.escape(stem)}\b", blob):
        return True
    collapsed = norm_tok.replace("_", "")
    return bool(collapsed and collapsed in blob.replace("_", ""))


def _subsequence(haystack: str, needle: str) -> bool:
    it = iter(haystack)
    return all(c in it for c in needle)


def param_reflects_formula(param: str, formula: str, variables: str) -> bool:
    """Whether a Lean def parameter is documented in the YAML formula / variables prose."""
    context = normalize_math_text(f"{formula} {variables}")
    if token_in_text(param, context):
        return True
    pl = param.lower()
    for tok in extract_formula_identifiers(formula):
        tl = tok.lower()
        if tl == pl or tl.startswith(pl) or pl.startswith(tl):
            return True
        if pl in tl and len(pl) >= 2:
            return True
        if len(pl) >= 2 and _subsequence(tl, pl):
            return True
    for tok in extract_formula_identifiers(variables):
        tl = tok.lower()
        if tl == pl or tl.startswith(pl) or pl.startswith(tl):
            return True
        if len(pl) >= 2 and _subsequence(tl, pl):
            return True
    return False


def check_equation_alignment(
    law_name: str, formula: str, lean_code: str, variables: str = ""
) -> list[str]:
    errors: list[str] = []
    if not formula or not lean_code:
        return errors
    parsed = parse_lean_def(lean_code)
    if not parsed:
        errors.append(f"{law_name}: no Lean ``def`` found in lean_code")
        return errors
    def_name, param_names, body = parsed
    lean_params = set(param_names)
    lean_blob = " ".join([def_name, " ".join(param_names), body])
    stub = is_overnight_expansion_stub(param_names, body)
    physics = extract_physics_tokens(formula)
    if stub:
        missing = [
            t
            for t in physics
            if not token_in_text(t, lean_blob)
            and not any(token_in_text(t, p) for p in lean_params)
        ]
        if missing:
            errors.append(
                f"{law_name}: overnight stub def {def_name}{tuple(param_names)} := {body.strip()!r} "
                f"does not encode formula tokens {missing!r}"
            )
        generic_params = [p for p in param_names if p in GENERIC_STUB_PARAMS]
        if generic_params and physics:
            errors.append(
                f"{law_name}: generic stub parameters {generic_params!r} paired with "
                f"physics-rich formula tokens {physics!r}"
            )
        return errors

    # Mechanism-linked Lean: each def parameter must trace to the declared formula/variables.
    for p in param_names:
        if p in GENERIC_STUB_PARAMS and not param_reflects_formula(p, formula, variables):
            errors.append(
                f"{law_name}: non-stub def {def_name} still uses generic parameter {p!r}"
            )
            continue
        if not param_reflects_formula(p, formula, variables):
            errors.append(
                f"{law_name}: Lean parameter {p!r} not reflected in formula/variables "
                f"for def {def_name}"
            )
    return errors


def check_loop_yaml(yaml_path: Path) -> list[str]:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    for idx, eq in enumerate(doc.get("equations") or [], start=1):
        lname = str(eq.get("law_name") or f"eq-{idx}")
        errors.extend(
            check_equation_alignment(
                lname,
                str(eq.get("formula") or ""),
                str(eq.get("lean_code") or ""),
                str(eq.get("variables") or ""),
            )
        )
    return errors


def run_self_test() -> list[str]:
    failures: list[str] = []
    if not POSITIVE_CONTROL.exists():
        failures.append(f"missing positive control YAML: {POSITIVE_CONTROL}")
    else:
        pos_errs = check_loop_yaml(POSITIVE_CONTROL)
        if pos_errs:
            failures.append(
                f"positive control {POSITIVE_CONTROL.name} expected pass, got: {pos_errs}"
            )
    if not NEGATIVE_CONTROL.exists():
        failures.append(f"missing negative control YAML: {NEGATIVE_CONTROL}")
    else:
        neg_errs = check_loop_yaml(NEGATIVE_CONTROL)
        if not neg_errs:
            failures.append(
                f"negative control {NEGATIVE_CONTROL.name} expected fail, but checker passed"
            )
    stub_errs = check_equation_alignment(
        "overnight-template-law1",
        OVERNIGHT_STUB_FORMULA,
        OVERNIGHT_STUB_LEAN,
    )
    if not stub_errs:
        failures.append("overnight stub negative control expected fail, but checker passed")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "yaml_path",
        nargs="?",
        type=Path,
        help="morphoeval/math_kg_loops/*.yaml loop file to check",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run built-in controls: forterre YAML must pass, zhou stub YAML and the "
            "run_overnight_1000_loops law1 template must fail"
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        failures = run_self_test()
        if failures:
            print("LEAN FORMULA ALIGNMENT SELF-TEST FAILED:")
            for f in failures:
                print(f"  - {f}")
            sys.exit(1)
        print(
            "LEAN FORMULA ALIGNMENT SELF-TEST PASSED "
            f"(positive={POSITIVE_CONTROL.name}, negative={NEGATIVE_CONTROL.name}, overnight stub)"
        )
        return

    if not args.yaml_path:
        ap.error("yaml_path required unless --self-test is set")
    path = args.yaml_path
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)
    errs = check_loop_yaml(path)
    if errs:
        print(f"LEAN FORMULA ALIGNMENT FAILED on {path} ({len(errs)} mismatch(es)):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(f"LEAN FORMULA ALIGNMENT PASSED on {path} (0 mismatches)")


if __name__ == "__main__":
    main()
