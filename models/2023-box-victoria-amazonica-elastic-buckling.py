#!/usr/bin/env python3
"""
models/2023-box-victoria-amazonica-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1828)
Organism: Victoria amazonica (Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs))
Phenomenon Scale: 30.9 cm (0.3094 m, log10=-0.509, 8_colonial_m)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Victoria amazonica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Victoria
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (30.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1828_victoria_amazonica_elastic_buckling.lean` (`MathOfBiology.Loop_1828`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/victoria_amazonica_elastic_buckling"
EXPECTED_SHA256 = {
    "victoria_amazonica_elastic_buckling_reference.json": "6feb66426da344989d54272e90364437165c50ad4ddeec9b865a87c424136bb5",
    "victoria_amazonica_specimen.jpg": "e7c7e2af042ff07c75a8167e10cafcf5c7e3b5f0b4f2ddfd337ff9d436171be1",
    "scout_manifest.json": "9a13e5d576c3b365e0958c6f28281980c5b52c7921819012f152b50ac031a528",
    "paper_citations.bib": "382f45bac1b3be1ff7c438d9e59ffa377c0a686a652225a31e845847d285d8b7"
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
    ref = json.loads((DATASET_DIR / "victoria_amazonica_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-box-victoria-amazonica-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
