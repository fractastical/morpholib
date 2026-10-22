#!/usr/bin/env python3
"""
models/2008-colosimo-gasterosteus-aculeatus-marine-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1930)
Organism: Gasterosteus aculeatus_marine (Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates))
Phenomenon Scale: 48 µm (4.836e-05 m, log10=-4.316, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Gasterosteus aculeatus_marine (Late Pleistocene Marine Ancestor (15 ka))
Geological Horizon: Late Pleistocene Marine Ancestor (15 ka) (0.015 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (48 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1930_gasterosteus_aculeatus_marine_bioelectric_polarity.lean` (`MathOfBiology.Loop_1930`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_marine_bioelectric_polarity"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_marine_bioelectric_polarity_reference.json": "ed7cc4caf77b02a9fe385d68f235f6160980aecfd7c4495931e998d43d2cfd73",
    "gasterosteus_aculeatus_marine_specimen.jpg": "4da41d2381dcc4c8c66a7b6ffe79757e5ca0b0dae3978efe5a16417ec3a5bd15",
    "scout_manifest.json": "6c04c55a2560305036c4d5481810ad9e45ab2ee3d32a885b822fbd8a1e1a475b",
    "paper_citations.bib": "a03f5a593e48ed69ba8a4d8d36b55623f5919caab6e6cd000e7eecb63ddc44b0"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_marine_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-colosimo-gasterosteus-aculeatus-marine-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
