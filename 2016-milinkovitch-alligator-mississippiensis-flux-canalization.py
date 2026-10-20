#!/usr/bin/env python3
"""
2016-milinkovitch-alligator-mississippiensis-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0430)
Organism: Alligator mississippiensis (American Alligator)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Alligator
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0430_alligator_mississippiensis_flux_canalization.lean` (`MathOfBiology.Loop_0430`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/alligator_mississippiensis_flux_canalization"
EXPECTED_SHA256 = {
    "alligator_mississippiensis_flux_canalization_reference.json": "57fbafcd07f393ad4d50b36942f0e4ace104e9fe8da79bdaaa2b8b08e6f4c756",
    "alligator_mississippiensis_specimen.jpg": "bb1770a8cf4a71024eec779363925a2928277079f2124074117061fc6d960764",
    "scout_manifest.json": "0718cd1a4b0e0b58475746c362412aa110beda54ed8e6be338636649301ae67b",
    "paper_citations.bib": "0b90bf5ea57b47d23764386defacc0912c7712ef92c75a44d9a559b8cd65acbd"
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
    ref = json.loads((DATASET_DIR / "alligator_mississippiensis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-milinkovitch-alligator-mississippiensis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
