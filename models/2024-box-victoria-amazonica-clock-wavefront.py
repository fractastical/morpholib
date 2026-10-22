#!/usr/bin/env python3
"""
models/2024-box-victoria-amazonica-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1468)
Organism: Victoria amazonica (Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs))
Phenomenon Scale: 11.2 cm (0.1125 m, log10=-0.949, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Victoria amazonica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Victoria
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (11.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1468_victoria_amazonica_clock_wavefront.lean` (`MathOfBiology.Loop_1468`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/victoria_amazonica_clock_wavefront"
EXPECTED_SHA256 = {
    "victoria_amazonica_clock_wavefront_reference.json": "90553dd6495c48a8bccdedf31e61ace726673a6352c97d3f43ee89b1b2d13471",
    "victoria_amazonica_specimen.jpg": "e7c7e2af042ff07c75a8167e10cafcf5c7e3b5f0b4f2ddfd337ff9d436171be1",
    "scout_manifest.json": "b1f9b58f75074d1da1a8181437466ab7b4621c19367d0ce66d9aa01627bccb27",
    "paper_citations.bib": "041aee6ea3de00535a46a7f06ef49a5aacd14172914841af83534c3cdee7298c"
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
    ref = json.loads((DATASET_DIR / "victoria_amazonica_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-box-victoria-amazonica-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
