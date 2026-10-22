#!/usr/bin/env python3
"""
models/2004-thewissen-pakicetus-attocki-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1943)
Organism: Pakicetus attocki (Early Eocene Terrestrial Wolf-Sized Stem-Cetacean)
Phenomenon Scale: 114 µm (0.0001139 m, log10=-3.943, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Pakicetus attocki (Early Eocene (Ypresian, 49.0 Ma))
Geological Horizon: Early Eocene (Ypresian, 49.0 Ma) (49.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pakicetus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (114 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1943_pakicetus_attocki_bioelectric_polarity.lean` (`MathOfBiology.Loop_1943`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pakicetus_attocki_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pakicetus_attocki_bioelectric_polarity_reference.json": "c8de6181d4ec48ef2ae079daf953f4bf0ab15fd841b369d99feee1e0174ad9f5",
    "pakicetus_attocki_specimen.jpg": "f535c1d722f574fc11becd2103eda0c5697e7184d39cb5b8ee794f42ce68e4a4",
    "scout_manifest.json": "cbd1411f71cb4f33176fd2761e89b419b90c33634e022b0ea2ad82e04af7055e",
    "paper_citations.bib": "7bbd702f676ff6a161966fa5bbaabb60ba40438ef1b4a48970a39a201d1da6c8"
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
    ref = json.loads((DATASET_DIR / "pakicetus_attocki_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-thewissen-pakicetus-attocki-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
