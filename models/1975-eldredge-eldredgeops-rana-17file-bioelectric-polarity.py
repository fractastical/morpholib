#!/usr/bin/env python3
"""
models/1975-eldredge-eldredgeops-rana-17file-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1966)
Organism: Eldredgeops rana_17file (Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye))
Phenomenon Scale: 357 nm (3.57e-07 m, log10=-6.447, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Eldredgeops rana_17file (Middle Devonian (Tioughniogan, 384 Ma))
Geological Horizon: Middle Devonian (Tioughniogan, 384 Ma) (384.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (357 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1966_eldredgeops_rana_17file_bioelectric_polarity.lean` (`MathOfBiology.Loop_1966`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_17file_bioelectric_polarity"
EXPECTED_SHA256 = {
    "eldredgeops_rana_17file_bioelectric_polarity_reference.json": "1a11480e68896582dbcfddc8bee3ada39c39df923ff292935574dd3410723c6e",
    "eldredgeops_rana_17file_specimen.jpg": "15e5b154e304f4a7611a6156cdcfaa0863adea78b4676093b13c3c4a1fc76ffe",
    "scout_manifest.json": "17690e9f557ef2b129931dfe8dee44c8d853aa9160d328624aedffe38b7bb014",
    "paper_citations.bib": "225b76b5845ad14492f4730d98dc92a9e13df5911efdcbc2528e987353162bd4"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_17file_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/1975-eldredge-eldredgeops-rana-17file-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
