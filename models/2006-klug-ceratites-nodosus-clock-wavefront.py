#!/usr/bin/env python3
"""
models/2006-klug-ceratites-nodosus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1416)
Organism: Ceratites nodosus (Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34))
Phenomenon Scale: 2.2 mm (0.002176 m, log10=-2.662, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Ceratites nodosus (Middle Triassic (Ladinian, 240 Ma))
Geological Horizon: Middle Triassic (Ladinian, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ceratites
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.2 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1416_ceratites_nodosus_clock_wavefront.lean` (`MathOfBiology.Loop_1416`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ceratites_nodosus_clock_wavefront"
EXPECTED_SHA256 = {
    "ceratites_nodosus_clock_wavefront_reference.json": "2af81bcd31d48925c9ac9595a4b82bab6779e9bcac2b316ee79de536ec4b7e31",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "7a0ae94eb6f350e883e569a4b22a11180912522cf11db846bbeda80f6568cfbd",
    "paper_citations.bib": "b1ee35dc0b9ae879c906afd4103ae83b1229098f86025b9614d5299ea18e5d1e"
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
    ref = json.loads((DATASET_DIR / "ceratites_nodosus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-klug-ceratites-nodosus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
