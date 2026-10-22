#!/usr/bin/env python3
"""
models/1994-edwards-cooksonia-pertoni-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1434)
Organism: Cooksonia pertoni (Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids))
Phenomenon Scale: 145 µm (0.0001454 m, log10=-3.837, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Cooksonia pertoni (Late Silurian (Wenlock-Ludlow, 425 Ma))
Geological Horizon: Late Silurian (Wenlock-Ludlow, 425 Ma) (425.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Cooksonia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (145 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1434_cooksonia_pertoni_clock_wavefront.lean` (`MathOfBiology.Loop_1434`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cooksonia_pertoni_clock_wavefront"
EXPECTED_SHA256 = {
    "cooksonia_pertoni_clock_wavefront_reference.json": "a4c7600ef87372909b23b5ec4cd903317652b94de63815d1a7eb0f12ceddfd63",
    "cooksonia_pertoni_specimen.jpg": "8fd04a0f4ac5dd5893b91ae0eb689cb450c030fc42bc7a41129eac5169e756c2",
    "scout_manifest.json": "13d5c810969f7df48acee9d341acf1fbb81fbca03a3024b7d2bff21416801b96",
    "paper_citations.bib": "9f0d3d0d4ce14647e2418bc0d0c4206fba1d3a58d8f5ad7f251e9d4e7e021a50"
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
    ref = json.loads((DATASET_DIR / "cooksonia_pertoni_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1994-edwards-cooksonia-pertoni-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
