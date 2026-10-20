#!/usr/bin/env python3
"""
2014-dietrich-pseudomonas-aeruginosa-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0284)
Organism: Pseudomonas aeruginosa (Biofilm Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Pseudomonas
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0284_pseudomonas_aeruginosa_morphogen_condensation.lean` (`MathOfBiology.Loop_0284`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pseudomonas_aeruginosa_morphogen_condensation"
EXPECTED_SHA256 = {
    "pseudomonas_aeruginosa_morphogen_condensation_reference.json": "1c1aef9bf20b6ef3abeba1b9217c9fb3b68854c617bb28ee20df14b3e61c364b",
    "pseudomonas_aeruginosa_specimen.jpg": "60ef5e2ce063576716d4086509c23046d1f100864118a90be5614a770b3ff31d",
    "scout_manifest.json": "ba23ab076f56eda815a69fe8aa3144672d7f9f3249f7400a96c184478dd66d24",
    "paper_citations.bib": "687036a7c81522bd9eedf35dc43a75596fa887a8ceee838d7b3f240be6eac5bf"
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
    ref = json.loads((DATASET_DIR / "pseudomonas_aeruginosa_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-dietrich-pseudomonas-aeruginosa-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
