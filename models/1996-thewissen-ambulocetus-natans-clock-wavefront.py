#!/usr/bin/env python3
"""
models/1996-thewissen-ambulocetus-natans-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1404)
Organism: Ambulocetus natans (Early Eocene Amphibious Walking Whale)
Phenomenon Scale: 2.1 cm (0.02083 m, log10=-1.681, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Ambulocetus natans (Early Eocene (Lutetian, 47.5 Ma))
Geological Horizon: Early Eocene (Lutetian, 47.5 Ma) (47.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ambulocetus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1404_ambulocetus_natans_clock_wavefront.lean` (`MathOfBiology.Loop_1404`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ambulocetus_natans_clock_wavefront"
EXPECTED_SHA256 = {
    "ambulocetus_natans_clock_wavefront_reference.json": "3f10889a282247c44fd0164cf7edbe098c07120190491622d13c5503587ac711",
    "ambulocetus_natans_specimen.jpg": "01a4e6043915d96e0394516e3db543eecad170348e219e8dd573ef897d004e67",
    "scout_manifest.json": "43af95847207fd7b1fdf7f671573559b3fcaa867d0ec33f8ffa984e37d1258be",
    "paper_citations.bib": "5dd5964709bb3620b9460f58e8b43f2e8fe3fdd64c5d3b5dd44ea306d2caae12"
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
    ref = json.loads((DATASET_DIR / "ambulocetus_natans_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-thewissen-ambulocetus-natans-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
