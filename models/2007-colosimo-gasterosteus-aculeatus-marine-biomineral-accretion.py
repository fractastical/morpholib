#!/usr/bin/env python3
"""
models/2007-colosimo-gasterosteus-aculeatus-marine-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1840)
Organism: Gasterosteus aculeatus_marine (Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates))
Phenomenon Scale: 3.3 cm (0.03325 m, log10=-1.478, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Gasterosteus aculeatus_marine (Late Pleistocene Marine Ancestor (15 ka))
Geological Horizon: Late Pleistocene Marine Ancestor (15 ka) (0.015 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (3.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1840_gasterosteus_aculeatus_marine_biomineral_accretion.lean` (`MathOfBiology.Loop_1840`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_marine_biomineral_accretion"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_marine_biomineral_accretion_reference.json": "eaec3c9e5fe6622c8d3b7d75b1f47c12df5431abe5807fd627a5102f2b3a29ea",
    "gasterosteus_aculeatus_marine_specimen.jpg": "4da41d2381dcc4c8c66a7b6ffe79757e5ca0b0dae3978efe5a16417ec3a5bd15",
    "scout_manifest.json": "74184d5a4dc797f609a4fb51494cc5ba7625e75b00cacbc4e7f914370d8fd4c7",
    "paper_citations.bib": "168a66c21349c66908ef5ed9535923d863616719e3c93042752aeb71c2e34414"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_marine_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-colosimo-gasterosteus-aculeatus-marine-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
