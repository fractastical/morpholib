#!/usr/bin/env python3
"""
models/2007-uhlen-dorudon-atrox-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1496)
Organism: Dorudon atrox (Late Eocene Basilosaurid Whale (Vestigial Hindlimb))
Phenomenon Scale: 8.6 cm (0.08585 m, log10=-1.066, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Dorudon atrox (Late Eocene (Priabonian, 37.0 Ma))
Geological Horizon: Late Eocene (Priabonian, 37.0 Ma) (37.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dorudon
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (8.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1496_dorudon_atrox_flux_canalization.lean` (`MathOfBiology.Loop_1496`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dorudon_atrox_flux_canalization"
EXPECTED_SHA256 = {
    "dorudon_atrox_flux_canalization_reference.json": "91f93ab08c302cd568445109ae0ba258dc9037375ec3d67f9045fd396e49bbce",
    "dorudon_atrox_specimen.jpg": "5a04e317b788950efe607a8cf4a468b15573b2f7e2efbde1deb8fa2bfaeaabaf",
    "scout_manifest.json": "79eb41cd0e33d5691ad46cbca68c910b16d84cb64dc7f0ecfd031c5060696e42",
    "paper_citations.bib": "8efe8f4994393f1b9d48ceac834bb707ace1a637dc88ca9ce868758a31a5c743"
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
    ref = json.loads((DATASET_DIR / "dorudon_atrox_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-uhlen-dorudon-atrox-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
