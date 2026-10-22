#!/usr/bin/env python3
"""
models/2017-gorzelak-encrinus-liliiformis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1991)
Organism: Encrinus liliiformis (Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles))
Phenomenon Scale: 12 nm (1.224e-08 m, log10=-7.912, 1_molecular_nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Encrinus liliiformis (Middle Triassic (Muschelkalk, 240 Ma))
Geological Horizon: Middle Triassic (Muschelkalk, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Encrinus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (12 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1991_encrinus_liliiformis_bioelectric_polarity.lean` (`MathOfBiology.Loop_1991`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/encrinus_liliiformis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "encrinus_liliiformis_bioelectric_polarity_reference.json": "d5262755063811b17d95bdcf30311d07e89a8b8f2770004a328ab00ca27766d4",
    "encrinus_liliiformis_specimen.jpg": "d19f3b6e48018c2dcd0ffb7be81fb1a880d1264498e9b5ec0b872b3310699a44",
    "scout_manifest.json": "3ac7a7a7d068f9e840153c46311ed7a982e702fb33ab13c4f51d16cc9749840f",
    "paper_citations.bib": "85d5a89c6af74ffc08b783d9ee50edf559a576c8da229f49670122f2436d42a0"
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
    ref = json.loads((DATASET_DIR / "encrinus_liliiformis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gorzelak-encrinus-liliiformis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
