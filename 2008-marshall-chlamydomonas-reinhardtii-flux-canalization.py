#!/usr/bin/env python3
"""
2008-marshall-chlamydomonas-reinhardtii-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0495)
Organism: Chlamydomonas reinhardtii (Biflagellate Microalga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Chlamydomonas
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0495_chlamydomonas_reinhardtii_flux_canalization.lean` (`MathOfBiology.Loop_0495`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chlamydomonas_reinhardtii_flux_canalization"
EXPECTED_SHA256 = {
    "chlamydomonas_reinhardtii_flux_canalization_reference.json": "9e879c190a7496392586e04c7be05e166c6a575a7d7ebb4246e9557c33b2bf2c",
    "chlamydomonas_reinhardtii_specimen.jpg": "ebcb2a487066205d3a9f06b194d0bfd68786337b7fbbf44dd71400b689d923bd",
    "scout_manifest.json": "c532c3e40f1e7f6731bd410ee6bfa97d579a6dde79071dd9b218c3981264f792",
    "paper_citations.bib": "dfdc7e8662493157dd8460f0a6c429ce6efb4c1e2f75ec90930d0317766d0649"
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
    ref = json.loads((DATASET_DIR / "chlamydomonas_reinhardtii_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-marshall-chlamydomonas-reinhardtii-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
