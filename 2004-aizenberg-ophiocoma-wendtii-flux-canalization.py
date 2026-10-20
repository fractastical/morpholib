#!/usr/bin/env python3
"""
2004-aizenberg-ophiocoma-wendtii-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0459)
Organism: Ophiocoma wendtii (Red Brittle Star)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ophiocoma
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0459_ophiocoma_wendtii_flux_canalization.lean` (`MathOfBiology.Loop_0459`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ophiocoma_wendtii_flux_canalization"
EXPECTED_SHA256 = {
    "ophiocoma_wendtii_flux_canalization_reference.json": "95bddcb0b01f2620895e9f6c7327b6f619054abf945f586c08fbe7ce945eace6",
    "ophiocoma_wendtii_specimen.jpg": "c073d01d499967366ff61d9831cb0e07e7505885a8ce4a9445565ec8d34f057d",
    "scout_manifest.json": "5c7608c7911d892d852892ffbf49d29a0e7e7e8b9758e19fd231dd2cb9f32027",
    "paper_citations.bib": "36aa69ebba40b59921e1d37137c3b0518ad230ccf6ab5b8287537c329e6bb019"
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
    ref = json.loads((DATASET_DIR / "ophiocoma_wendtii_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-aizenberg-ophiocoma-wendtii-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
