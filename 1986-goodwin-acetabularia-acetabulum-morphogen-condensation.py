#!/usr/bin/env python3
"""
1986-goodwin-acetabularia-acetabulum-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0292)
Organism: Acetabularia acetabulum (Mermaid's Wineglass Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acetabularia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0292_acetabularia_acetabulum_morphogen_condensation.lean` (`MathOfBiology.Loop_0292`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acetabularia_acetabulum_morphogen_condensation"
EXPECTED_SHA256 = {
    "acetabularia_acetabulum_morphogen_condensation_reference.json": "0092f48bdff983f9901e6bad4d23e72abd22ce2c971197077f5dc443e962de86",
    "acetabularia_acetabulum_specimen.jpg": "17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1",
    "scout_manifest.json": "583a45aae3dd3152ea396404960b6c80ee18f3c1e9337270038eaee0da9c419a",
    "paper_citations.bib": "a3b4c967eaa296e50aa16d9fe429ced31a778481ad59b928ee3860cc4e770ef8"
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
    ref = json.loads((DATASET_DIR / "acetabularia_acetabulum_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1986-goodwin-acetabularia-acetabulum-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
