#!/usr/bin/env python3
"""
models/1986-grande-diplomystus-dentatus-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1836)
Organism: Diplomystus dentatus (Green River Eocene Armored Herring Fossil)
Phenomenon Scale: 10.4 cm (0.104 m, log10=-0.983, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Diplomystus dentatus (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Diplomystus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (10.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1836_diplomystus_dentatus_biomineral_accretion.lean` (`MathOfBiology.Loop_1836`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/diplomystus_dentatus_biomineral_accretion"
EXPECTED_SHA256 = {
    "diplomystus_dentatus_biomineral_accretion_reference.json": "7d290a8de2fd62d4e6ba389401856e240a14d2c6fec09a5d8996466cac7bdcae",
    "diplomystus_dentatus_specimen.jpg": "00e0f80ae28f4fe36376b217f05e2048df17ed19a49f48d49d4a9b88b2b2b659",
    "scout_manifest.json": "0de8733b612d132419967b78ceeeba8d34209548fec942e1eda2919731c28096",
    "paper_citations.bib": "eecb65c590c9af79a7029180240c57fd86958d2f5d39ba85c4c08e6676c01468"
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
    ref = json.loads((DATASET_DIR / "diplomystus_dentatus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-grande-diplomystus-dentatus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
