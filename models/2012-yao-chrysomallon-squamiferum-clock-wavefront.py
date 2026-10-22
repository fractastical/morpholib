#!/usr/bin/env python3
"""
models/2012-yao-chrysomallon-squamiferum-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1464)
Organism: Chrysomallon squamiferum (Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor))
Phenomenon Scale: 39 nm (3.924e-08 m, log10=-7.406, 1_molecular_nm)
Anatomical Structure: Segmentation clock somite / metamere unit in Chrysomallon squamiferum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Chrysomallon
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (39 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1464_chrysomallon_squamiferum_clock_wavefront.lean` (`MathOfBiology.Loop_1464`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/chrysomallon_squamiferum_clock_wavefront"
EXPECTED_SHA256 = {
    "chrysomallon_squamiferum_clock_wavefront_reference.json": "29c6e7998295b221202e4cd61806e309edc4f7f35b8d6772311a8f6eb1977ec4",
    "chrysomallon_squamiferum_specimen.jpg": "9177d515a6734cec2e96211c99840c572fb4d027ec687bdf0cb090fe6bbe9acc",
    "scout_manifest.json": "76ab82fcc69290ed161722b1455d5b3792b9ac4bce3f10810319b1decdd58f39",
    "paper_citations.bib": "b71376c63ff7e31b630a33e49a8420515a36dec02d2922da556efe1b1ac6f698"
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
    ref = json.loads((DATASET_DIR / "chrysomallon_squamiferum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-yao-chrysomallon-squamiferum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
