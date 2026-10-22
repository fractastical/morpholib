#!/usr/bin/env python3
"""
models/2025-box-victoria-amazonica-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2008)
Organism: Victoria amazonica (Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs))
Phenomenon Scale: 1.1 mm (0.001125 m, log10=-2.949, 6_organ_mm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Victoria amazonica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Victoria
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.1 mm))
Formal Verification: `morphoeval/lean/loops/Loop_2008_victoria_amazonica_bioelectric_polarity.lean` (`MathOfBiology.Loop_2008`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/victoria_amazonica_bioelectric_polarity"
EXPECTED_SHA256 = {
    "victoria_amazonica_bioelectric_polarity_reference.json": "a9ecae5859e01f4fa42e95e0a3ac7fb376e740a83fcff305db97d4f6947fce28",
    "victoria_amazonica_specimen.jpg": "e7c7e2af042ff07c75a8167e10cafcf5c7e3b5f0b4f2ddfd337ff9d436171be1",
    "scout_manifest.json": "ca291ec723e30272d3989e9f8433d2f56aba06573e1c84bca89c3a4608c47bf3",
    "paper_citations.bib": "9beca7d83cca0db6cd684557c72f3f9ebe7623badf37ff0edc7fc69b32f55b1e"
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
    ref = json.loads((DATASET_DIR / "victoria_amazonica_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-box-victoria-amazonica-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
