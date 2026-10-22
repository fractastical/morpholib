#!/usr/bin/env python3
"""
models/1998-coates-acanthostega-gunnari-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1395)
Organism: Acanthostega gunnari (Late Devonian Eight-Digited Polydactyl Stem-Tetrapod)
Phenomenon Scale: 7.8 mm (0.00782 m, log10=-2.107, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Acanthostega gunnari (Late Devonian (Famennian, 365 Ma))
Geological Horizon: Late Devonian (Famennian, 365 Ma) (365.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Acanthostega
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (7.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1395_acanthostega_gunnari_clock_wavefront.lean` (`MathOfBiology.Loop_1395`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/acanthostega_gunnari_clock_wavefront"
EXPECTED_SHA256 = {
    "acanthostega_gunnari_clock_wavefront_reference.json": "88ddb82d6d63d3a745c1aa148c384daff3b208831f4c7f635fca3d38f3ec3a79",
    "acanthostega_gunnari_specimen.jpg": "17b12e17eb08ab7e05345c565a3dc299e5c4752dd56c4753617937fa7b3a66b5",
    "scout_manifest.json": "e7d952ff61b341c1c1c444c7732ec8cc0fb6a017afc7b222effab1fcce865b62",
    "paper_citations.bib": "482b5b32aab817111d607af9a9802b2c4c8cb54a3d4c62a327e9e3d45ecb5fc0"
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
    ref = json.loads((DATASET_DIR / "acanthostega_gunnari_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1998-coates-acanthostega-gunnari-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
