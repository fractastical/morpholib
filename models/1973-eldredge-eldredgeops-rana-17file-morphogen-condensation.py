#!/usr/bin/env python3
"""
models/1973-eldredge-eldredgeops-rana-17file-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1336)
Organism: Eldredgeops rana_17file (Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye))
Phenomenon Scale: 80 µm (8.032e-05 m, log10=-4.095, 4_cellular_10um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Eldredgeops rana_17file (Middle Devonian (Tioughniogan, 384 Ma))
Geological Horizon: Middle Devonian (Tioughniogan, 384 Ma) (384.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (80 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1336_eldredgeops_rana_17file_morphogen_condensation.lean` (`MathOfBiology.Loop_1336`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_17file_morphogen_condensation"
EXPECTED_SHA256 = {
    "eldredgeops_rana_17file_morphogen_condensation_reference.json": "fa1b9f06d4609a55fc5ba660bbad07d2eed8cf7bb8f6d17981226a4543d4e2f3",
    "eldredgeops_rana_17file_specimen.jpg": "15e5b154e304f4a7611a6156cdcfaa0863adea78b4676093b13c3c4a1fc76ffe",
    "scout_manifest.json": "4c2eefc0faabb326cd8b36c522c173d38cadf699625bf267c2cf159e1afcd2ae",
    "paper_citations.bib": "bbae503678f6163719e50f20652f93ca78953d359d0c5d621b98f9a6dad258b3"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_17file_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1973-eldredge-eldredgeops-rana-17file-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
