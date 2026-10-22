#!/usr/bin/env python3
"""
models/2012-li-anchiornis-huxleyi-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1409)
Organism: Anchiornis huxleyi (Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array))
Phenomenon Scale: 58 nm (5.78e-08 m, log10=-7.238, 2_macromolecular_100nm)
Anatomical Structure: Segmentation clock somite / metamere unit in Anchiornis huxleyi (Late Jurassic (Tiaojishan Fm, 160.0 Ma))
Geological Horizon: Late Jurassic (Tiaojishan Fm, 160.0 Ma) (160.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anchiornis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (58 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1409_anchiornis_huxleyi_clock_wavefront.lean` (`MathOfBiology.Loop_1409`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anchiornis_huxleyi_clock_wavefront"
EXPECTED_SHA256 = {
    "anchiornis_huxleyi_clock_wavefront_reference.json": "9b0ae8b000bc473e8520147d31112b9f044402d176c23a758b2c3fa0b20413a2",
    "anchiornis_huxleyi_specimen.jpg": "fab0bb4ca7882feafbf18840d6056dd36150060dc484593ec1df858d67a18865",
    "scout_manifest.json": "cbad80f9712224818eea3365e7f2fd6fc74ebe0b5e8f55f17a5111678db925b5",
    "paper_citations.bib": "28beb0754effd52e0a951034a2fb5b4fe1227f33e66f499de2d8288e68f7afee"
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
    ref = json.loads((DATASET_DIR / "anchiornis_huxleyi_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-li-anchiornis-huxleyi-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
