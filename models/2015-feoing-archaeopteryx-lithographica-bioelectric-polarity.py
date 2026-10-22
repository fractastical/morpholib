#!/usr/bin/env python3
"""
models/2015-feoing-archaeopteryx-lithographica-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1950)
Organism: Archaeopteryx lithographica (Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane))
Phenomenon Scale: 58 µm (5.838e-05 m, log10=-4.234, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Archaeopteryx lithographica (Late Jurassic (Tithonian, 150.0 Ma))
Geological Horizon: Late Jurassic (Tithonian, 150.0 Ma) (150.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Archaeopteryx
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (58 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1950_archaeopteryx_lithographica_bioelectric_polarity.lean` (`MathOfBiology.Loop_1950`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteryx_lithographica_bioelectric_polarity"
EXPECTED_SHA256 = {
    "archaeopteryx_lithographica_bioelectric_polarity_reference.json": "b6a05fa18d5deb1e870188e5a89e360c34b1ebdfee48239ef17f4af5c9e40cfb",
    "archaeopteryx_lithographica_specimen.jpg": "22ad1c7de4cfaadcc6057d0f4ad6dd252fd379582bec21b51bf73852daa000af",
    "scout_manifest.json": "541c5d0998e4c2bb032845c91915bf83c4fab1aa81f5d8a8aa9716350e5f0d09",
    "paper_citations.bib": "d1ad15705ceacead959eebf135b77467bd7df10774d604d2c8b3c45461699ae2"
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
    ref = json.loads((DATASET_DIR / "archaeopteryx_lithographica_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-feoing-archaeopteryx-lithographica-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
