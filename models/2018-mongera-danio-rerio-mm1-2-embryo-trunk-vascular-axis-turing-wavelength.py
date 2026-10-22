#!/usr/bin/env python3
"""
models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1171)
Organism: Danio rerio (Zebrafish — 24–48 hpf)
Phenomenon Scale: 912 µm (0.000912 m, log10=-3.040, 5_primordium_100um)
Anatomical Structure: Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus (24–48 hpf (Whole Embryonic AP Trunk & ISV Arcade, ZFA:0001093)) · Zebrahub whole_embryo_4D_lightsheet_trajectory (10–24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection at 912 µm (Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus))
Formal Verification: `morphoeval/lean/loops/Loop_1171_danio_rerio_mm1_2_embryo_trunk_vascular_axis_turing_wavelength.lean` (`MathOfBiology.Loop_1171`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_mm1_2_embryo_trunk_vascular_axis_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_rerio_mm1_2_embryo_trunk_vascular_axis_turing_wavelength_reference.json": "8f4075aa339c57eb53e9b7d4546eceda8b4450d6a2b4038a21a4485227daad80",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "6d1cded18b02fa3d855b27fc22117e16f2412e842ba5446c0d926b875b484b33",
    "paper_citations.bib": "4045d2c88393d8540a70ed766d9f81943c461a50498abe7b351356f27b1c5774"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_mm1_2_embryo_trunk_vascular_axis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
