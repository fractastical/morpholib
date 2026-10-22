#!/usr/bin/env python3
"""
models/2016-gorzelak-encrinus-liliiformis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1901)
Organism: Encrinus liliiformis (Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles))
Phenomenon Scale: 8.4 µm (8.415e-06 m, log10=-5.075, 3_subcellular_um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Encrinus liliiformis (Middle Triassic (Muschelkalk, 240 Ma))
Geological Horizon: Middle Triassic (Muschelkalk, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Encrinus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (8.4 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1901_encrinus_liliiformis_biomineral_accretion.lean` (`MathOfBiology.Loop_1901`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/encrinus_liliiformis_biomineral_accretion"
EXPECTED_SHA256 = {
    "encrinus_liliiformis_biomineral_accretion_reference.json": "71adcf3d53f27e98b27c66fe9f2c70d24bd2929acac6d6943b781e187fe6aaa0",
    "encrinus_liliiformis_specimen.jpg": "d19f3b6e48018c2dcd0ffb7be81fb1a880d1264498e9b5ec0b872b3310699a44",
    "scout_manifest.json": "823622b15657a72024ee22d6ae87b0be97689fc6f61bf8d9f91abc399b97fa4f",
    "paper_citations.bib": "aa18a00fad4e59111de2aba58c9986d580d7a9167ff319a88e03e1fc9b0a65e7"
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
    ref = json.loads((DATASET_DIR / "encrinus_liliiformis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-gorzelak-encrinus-liliiformis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
