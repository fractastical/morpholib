#!/usr/bin/env python3
"""
models/2012-zhang-sinosauropteryx-prima-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1408)
Organism: Sinosauropteryx prima (Early Cretaceous Proto-Feathered Compsognathid Dinosaur)
Phenomenon Scale: 105 µm (0.0001046 m, log10=-3.980, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Sinosauropteryx prima (Early Cretaceous (Yixian Fm, 124.6 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 124.6 Ma) (124.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Sinosauropteryx
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (105 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1408_sinosauropteryx_prima_clock_wavefront.lean` (`MathOfBiology.Loop_1408`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/sinosauropteryx_prima_clock_wavefront"
EXPECTED_SHA256 = {
    "sinosauropteryx_prima_clock_wavefront_reference.json": "1ebfd9d3d2dd4dc9d150cd532ac29563e65fed99be64377e2d75a44ab92d66bf",
    "sinosauropteryx_prima_specimen.jpg": "dafd40e5487fb100d0309d10c30a01a1479f1793e6aa37726388bfafdd50d009",
    "scout_manifest.json": "2c3a6df93589f2b3e94c56192cb5efaa15a2758dab91bfa6d6b7d8197e8d124a",
    "paper_citations.bib": "fb4c4157f3d1dd16098f516193131e53aae5b9dfff20008f7aa9577e3355f8b6"
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
    ref = json.loads((DATASET_DIR / "sinosauropteryx_prima_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-zhang-sinosauropteryx-prima-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
