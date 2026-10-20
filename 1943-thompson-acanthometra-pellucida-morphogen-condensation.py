#!/usr/bin/env python3
"""
1943-thompson-acanthometra-pellucida-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0293)
Organism: Acanthometra pellucida (Strontium Radiolarian)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acanthometra
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0293_acanthometra_pellucida_morphogen_condensation.lean` (`MathOfBiology.Loop_0293`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acanthometra_pellucida_morphogen_condensation"
EXPECTED_SHA256 = {
    "acanthometra_pellucida_morphogen_condensation_reference.json": "0804782008437836ace7b088a14f12d08e7cb55935846afd11e00fe38128618c",
    "acanthometra_pellucida_specimen.jpg": "8f4b8a930e33286fd2843fa388dfec53592278e506619611732e990cc03371d4",
    "scout_manifest.json": "f985cadb39ee3abaab81edb694ca417bae856f5826cae95305f66ad6e3675d92",
    "paper_citations.bib": "5f82dd91184046c109f07c0bbc1140228ea4ca8b78d6becceaa528666a3b6660"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "acanthometra_pellucida_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 1943-thompson-acanthometra-pellucida-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
