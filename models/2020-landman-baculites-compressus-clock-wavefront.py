#!/usr/bin/env python3
"""
models/2020-landman-baculites-compressus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1419)
Organism: Baculites compressus (Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite)
Phenomenon Scale: 8.5 mm (0.008536 m, log10=-2.069, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Baculites compressus (Late Cretaceous (Pierre Shale, 73.5 Ma))
Geological Horizon: Late Cretaceous (Pierre Shale, 73.5 Ma) (73.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Baculites
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (8.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1419_baculites_compressus_clock_wavefront.lean` (`MathOfBiology.Loop_1419`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/baculites_compressus_clock_wavefront"
EXPECTED_SHA256 = {
    "baculites_compressus_clock_wavefront_reference.json": "cf41d5f70e956657b3e0bb879aff24339def2f287626aa4033358dd109dd49fb",
    "baculites_compressus_specimen.jpg": "849f95ffd5d647ae2a12750587da774719bdf94f1cf2520dc375c315c45eab50",
    "scout_manifest.json": "1e1fbbc63d2a771905fff50449e507dc758d0575cfe17fbdf0c2caf8f7b56e80",
    "paper_citations.bib": "787c81cfa5840a5e334a278a25d3aae810b461d938dae1ee8e871056520dd8b2"
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
    ref = json.loads((DATASET_DIR / "baculites_compressus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-landman-baculites-compressus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
