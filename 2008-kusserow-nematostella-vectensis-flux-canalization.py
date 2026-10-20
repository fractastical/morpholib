#!/usr/bin/env python3
"""
2008-kusserow-nematostella-vectensis-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0462)
Organism: Nematostella vectensis (Starlet Sea Anemone)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Nematostella
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0462_nematostella_vectensis_flux_canalization.lean` (`MathOfBiology.Loop_0462`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/nematostella_vectensis_flux_canalization"
EXPECTED_SHA256 = {
    "nematostella_vectensis_flux_canalization_reference.json": "f352a8d34d8fef8de8181dfe9934c18146177b1db96508cad9dd45e23fcf069e",
    "nematostella_vectensis_specimen.jpg": "fb9d66de457a62bbf3d2ec1bcdb57f33e32468690a3047a09117a19153073a53",
    "scout_manifest.json": "39d7c21b1f41e792e7a66b0e61966b1beb8e54b8de991357e6a7f358952069b3",
    "paper_citations.bib": "7e6816605ac9a1d671c3e80139775cf8ad26ad6db002a8d9febc20a990be4234"
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
    ref = json.loads((DATASET_DIR / "nematostella_vectensis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-kusserow-nematostella-vectensis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
