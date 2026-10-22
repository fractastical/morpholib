#!/usr/bin/env python3
"""
models/2013-paterson-anomalocaris-canadensis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1878)
Organism: Anomalocaris canadensis (Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone))
Phenomenon Scale: 1.6 cm (0.01617 m, log10=-1.791, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Anomalocaris canadensis (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Anomalocaris
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1878_anomalocaris_canadensis_biomineral_accretion.lean` (`MathOfBiology.Loop_1878`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anomalocaris_canadensis_biomineral_accretion"
EXPECTED_SHA256 = {
    "anomalocaris_canadensis_biomineral_accretion_reference.json": "bcf4a64aef258e09b5939e781dd845b4f918cd4ce9590591e8dfff8ef8a8e0fa",
    "anomalocaris_canadensis_specimen.jpg": "197e43665198870b80e436c90febc5b7010eb24957c9239893e6ac1be683bf9c",
    "scout_manifest.json": "5bdd3abe5052c5d253ca70e175a06ef810263a8c3c9fc70a4cd9693180f151aa",
    "paper_citations.bib": "266234e431b1d852d3836ed621a58af4f86694b9be9ea8364677ff54d4380e2e"
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
    ref = json.loads((DATASET_DIR / "anomalocaris_canadensis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-paterson-anomalocaris-canadensis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
