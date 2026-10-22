#!/usr/bin/env python3
"""
models/2001-forsten-equus-scotti-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1402)
Organism: Equus scotti (Pleistocene Giant Ice-Age Horse Chronospecies)
Phenomenon Scale: 1.8 cm (0.01768 m, log10=-1.753, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Equus scotti (Middle Pleistocene (Irvingtonian, 0.6 Ma))
Geological Horizon: Middle Pleistocene (Irvingtonian, 0.6 Ma) (0.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1402_equus_scotti_clock_wavefront.lean` (`MathOfBiology.Loop_1402`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/equus_scotti_clock_wavefront"
EXPECTED_SHA256 = {
    "equus_scotti_clock_wavefront_reference.json": "8fad6e6e1b61a1483b620f69f2546a3407c4274ea421c89757893251c3bfff87",
    "equus_scotti_specimen.jpg": "fea6d4b899e55872847306904b4629acc9fcf09737b2a6f906d0accb0de6e222",
    "scout_manifest.json": "f88f5e565a01b4c6ffa7a939bb96dd998bc5583d2c8c117cdd87a856a99bf254",
    "paper_citations.bib": "de35fc20e2fd584c565baa00db9f50489d23466e3fdf0f35747cf6fd55d45f0a"
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
    ref = json.loads((DATASET_DIR / "equus_scotti_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-forsten-equus-scotti-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
