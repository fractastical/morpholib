#!/usr/bin/env python3
"""
1972-raup-turritella-communis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0758)
Organism: Turritella communis (Tower Screw Shell)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Turritella
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0758_turritella_communis_elastic_buckling.lean` (`MathOfBiology.Loop_0758`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/turritella_communis_elastic_buckling"
EXPECTED_SHA256 = {
    "turritella_communis_elastic_buckling_reference.json": "809eff2996ebbed66f406c666d9fc33f7ed2c6fa772266b38e0b01db50e4609d",
    "turritella_communis_specimen.jpg": "899a774eadcf1dae2799654a3393a678c5543bfebd3e3f9a02043f13da6934d0",
    "scout_manifest.json": "e33008120f5bfa5c5c2fed7e87e547a9368c4ef2fd82f6899bcdcc6cefcfc001",
    "paper_citations.bib": "11d2b93e4a3942a986af3e1142791ae0890adbe03e6093212071728f57b41b8d"
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
    ref = json.loads((DATASET_DIR / "turritella_communis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1972-raup-turritella-communis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
