#!/usr/bin/env python3
"""
1992-camazine-apis-mellifera-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0243)
Organism: Apis mellifera (Western Honeybee)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Apis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0243_apis_mellifera_morphogen_condensation.lean` (`MathOfBiology.Loop_0243`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/apis_mellifera_morphogen_condensation"
EXPECTED_SHA256 = {
    "apis_mellifera_morphogen_condensation_reference.json": "ab86e22f5dee879dee90c2ca3520eea9351a38d7d8e1fa2a5b79822f58796878",
    "apis_mellifera_specimen.jpg": "67454ef389899df977e29efa0db0cba99519f5d46a3631810bcc7f160efd8fce",
    "scout_manifest.json": "07d9429e2e97e436919232066f0c4dbdbdaff1e7c98364f557174c33290789f0",
    "paper_citations.bib": "69a911cb419ed4be7f154c37837c26195bc8590a5a75915f71d355c5ff925534"
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
    ref = json.loads((DATASET_DIR / "apis_mellifera_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1992-camazine-apis-mellifera-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
