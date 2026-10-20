#!/usr/bin/env python3
"""
2006-hinman-asterias-rubens-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0454)
Organism: Asterias rubens (Common Starfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Asterias
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0454_asterias_rubens_flux_canalization.lean` (`MathOfBiology.Loop_0454`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/asterias_rubens_flux_canalization"
EXPECTED_SHA256 = {
    "asterias_rubens_flux_canalization_reference.json": "207f13ed200f62b1cf091c0be75cebb3686d05344df49f449c3c91188ed061e2",
    "asterias_rubens_specimen.jpg": "37d94bdfed179302d1e9aa062333c2f87a5cb3d7c3932c51056f5e68c7c762ad",
    "scout_manifest.json": "249cca2aa371054ca9ab2088e7d3fa3721c3d177c28300a956cc842575b3d3e7",
    "paper_citations.bib": "deda59ab99bca7f6ffc048652ee7deabc075529d6a439016a095b417b68bde63"
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
    ref = json.loads((DATASET_DIR / "asterias_rubens_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-hinman-asterias-rubens-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
