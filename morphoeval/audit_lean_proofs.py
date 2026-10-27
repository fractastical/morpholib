#!/usr/bin/env python3
"""
morphoeval/audit_lean_proofs.py — Inventory the Lean template-placeholder problem.

Every overnight generator stamped each loop with the same 3 placeholder
theorems (`a*b > 0`, `p+q-q = p`, `N > M by decide`) while the matching
`math_kg_loops/*.yaml` registered them as formalizing the loop's real
governing law. This audit classifies every Lean loop + KG spec so the
clog can be measured and re-formalization work can be targeted:

  template_trivial   all 3 template declarations present (claims nothing real)
  template_partial   1-2 template declarations present
  substantive        no template declarations (law-specific formalization)
  no_lean_link       KG spec with no lean_decl/lean_code at all

The KG-side fix already lives in `math_kg.is_template_lean_equation`
(template equations are filtered from `load_dynamic_math_laws_registry`);
this script is the measurement/reporting side. It never edits proofs.

Usage:
  venv/bin/python morphoeval/audit_lean_proofs.py [--out morphoeval/runs/lean_proof_audit.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TEMPLATE_DEFS = (
    "def law1_invariant (a b : Int) : Int := a * b",
    "def law2_balance (p q : Int) : Int := p + q - q",
    "def law3_threshold_margin (lhs rhs : Int) : Int := lhs - rhs",
)


def classify_lean_text(text: str) -> tuple[str, int]:
    n = sum(1 for d in TEMPLATE_DEFS if d in text)
    return ("template_trivial" if n == 3 else "template_partial" if n else "substantive"), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "runs" / "lean_proof_audit.json"))
    args = ap.parse_args()

    lean_counts: dict[str, int] = {"template_trivial": 0, "template_partial": 0, "substantive": 0}
    lean_examples: dict[str, list[str]] = {k: [] for k in lean_counts}
    loops_dir = HERE / "lean" / "loops"
    lean_files = sorted(loops_dir.glob("*.lean")) if loops_dir.exists() else []
    for lf in lean_files:
        cls, _ = classify_lean_text(lf.read_text(encoding="utf-8", errors="replace"))
        lean_counts[cls] += 1
        if len(lean_examples[cls]) < 5:
            lean_examples[cls].append(str(lf.relative_to(ROOT)))

    kg_counts: dict[str, int] = {"template_only": 0, "has_substantive": 0, "no_lean_link": 0}
    n_template_eqs = n_substantive_eqs = 0
    kg_dir = HERE / "math_kg_loops"
    spec_files = sorted(kg_dir.glob("*.yaml")) if kg_dir.exists() else []
    for yf in spec_files:
        try:
            doc = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        eqs = doc.get("equations", []) or []
        templ = 0
        for e in eqs:
            if not isinstance(e, dict):
                continue
            code = str(e.get("lean_code", ""))
            decl = str(e.get("lean_decl", ""))
            if any(d in code for d in TEMPLATE_DEFS) or decl.endswith((".law1_invariant", ".law2_balance", ".law3_threshold_margin")) \
                    or e.get("formalization_status") == "template_placeholder_not_a_proof":
                templ += 1
            elif decl or code:
                n_substantive_eqs += 1
        n_template_eqs += templ
        if eqs and templ == len(eqs):
            kg_counts["template_only"] += 1
        elif eqs:
            kg_counts["has_substantive"] += 1
        else:
            kg_counts["no_lean_link"] += 1

    report: dict[str, Any] = {
        "problem": "Overnight generators emit identical placeholder Lean theorems (a*b>0 by Int.mul_pos; p+q-q=p by omega; numeral>numeral by decide) while math_kg_loops specs register them as formalizing each loop's named governing law. Valid Lean, vacuous claims, ~3 KG equation entries per loop.",
        "lean_loop_files": len(lean_files),
        "lean_classification": lean_counts,
        "lean_examples": lean_examples,
        "kg_spec_files": len(spec_files),
        "kg_classification": kg_counts,
        "kg_template_equations_blocked_by_gate": n_template_eqs,
        "kg_substantive_equations_admitted": n_substantive_eqs,
        "gate": "morphoeval/math_kg.py::is_template_lean_equation (applied in load_dynamic_math_laws_registry)",
        "remediation_path": [
            "1. DONE: KG gate filters template equations; template-only specs no longer register laws.",
            "2. Rebuild KG (build_morphoeval_kg.py) so published graphs drop the false formalization triples.",
            "3. Real fix per high-value loop: replace the template with a law-specific Lean formalization (state variables + the actual invariant/stability theorem), then its spec re-enters the KG automatically.",
            "4. Generators must stop emitting the template as a law claim (emit formalization_status: template_placeholder_not_a_proof, which the gate also blocks).",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
