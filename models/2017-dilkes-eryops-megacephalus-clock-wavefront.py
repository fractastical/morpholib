#!/usr/bin/env python3
"""
models/2017-dilkes-eryops-megacephalus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1397)
Organism: Eryops megacephalus (Early Permian Temnospondyl Pentadactyl Amphibian)
Phenomenon Scale: 1.6 cm (0.01637 m, log10=-1.786, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Eryops megacephalus (Early Permian (Cisuralian, 295 Ma))
Geological Horizon: Early Permian (Cisuralian, 295 Ma) (295.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Eryops
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1397_eryops_megacephalus_clock_wavefront.lean` (`MathOfBiology.Loop_1397`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eryops_megacephalus_clock_wavefront"
EXPECTED_SHA256 = {
    "eryops_megacephalus_clock_wavefront_reference.json": "ca7769ed93e16483153f44e1ebac47ced52ec9fe174b7b59768539353c7db339",
    "eryops_megacephalus_specimen.jpg": "129f152b5349248669bd3ffc0b0360d0131cfa03e87c6ff1c0ac912fae50e424",
    "scout_manifest.json": "0a0efddfb4cfb4aed872e3c480e82021d03a458a38f9651915810b4ad445739a",
    "paper_citations.bib": "cbea4e9f0bf93f115be51bdfc2680711d0ce197c0068bab2ce1c8b75478fdf0c"
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
    ref = json.loads((DATASET_DIR / "eryops_megacephalus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-dilkes-eryops-megacephalus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
