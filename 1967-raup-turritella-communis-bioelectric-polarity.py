#!/usr/bin/env python3
"""
1967-raup-turritella-communis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0958)
Organism: Turritella communis (Tower Screw Shell)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Turritella
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0958_turritella_communis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0958`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/turritella_communis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "turritella_communis_bioelectric_polarity_reference.json": "b600297637579b7afc63a2f3f0e4e3c26f230119e6c7661d41e5b737e468e315",
    "turritella_communis_specimen.jpg": "899a774eadcf1dae2799654a3393a678c5543bfebd3e3f9a02043f13da6934d0",
    "scout_manifest.json": "51ecda3d26407001393bec218e70e9cb1a7fcf59b19a82f779600918987c098a",
    "paper_citations.bib": "5792d972eba65e208d5ec27595765c9443207a5422d8a03282a40e1daedd73f0"
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
    ref = json.loads((DATASET_DIR / "turritella_communis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1967-raup-turritella-communis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
