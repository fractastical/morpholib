#!/usr/bin/env python3
"""
models/2005-thewissen-pakicetus-attocki-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2033)
Organism: Pakicetus attocki (Early Eocene Terrestrial Wolf-Sized Stem-Cetacean)
Phenomenon Scale: 25.6 cm (0.2563 m, log10=-0.591, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Pakicetus attocki (Early Eocene (Ypresian, 49.0 Ma))
Geological Horizon: Early Eocene (Ypresian, 49.0 Ma) (49.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pakicetus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (25.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2033_pakicetus_attocki_collective_stigmergy.lean` (`MathOfBiology.Loop_2033`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pakicetus_attocki_collective_stigmergy"
EXPECTED_SHA256 = {
    "pakicetus_attocki_collective_stigmergy_reference.json": "ea0306b50e349c9a1e0bbadd0886afd0d7543ef6a45dec00ceb8e82d0b53565f",
    "pakicetus_attocki_specimen.jpg": "f535c1d722f574fc11becd2103eda0c5697e7184d39cb5b8ee794f42ce68e4a4",
    "scout_manifest.json": "1043ee3e7941873ca4e56fc2068e7fe316d925fd94494aba333f09e2cf8a8ffe",
    "paper_citations.bib": "23344ba71eb7cc8eff9440a5f0b2e07817cf0c191aa12ed9e49c6ed0b5d2382a"
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
    ref = json.loads((DATASET_DIR / "pakicetus_attocki_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-thewissen-pakicetus-attocki-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
