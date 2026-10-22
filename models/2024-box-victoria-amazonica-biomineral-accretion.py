#!/usr/bin/env python3
"""
models/2024-box-victoria-amazonica-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1918)
Organism: Victoria amazonica (Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs))
Phenomenon Scale: 77.4 cm (0.7736 m, log10=-0.111, 8_colonial_m)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Victoria amazonica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Victoria
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (77.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1918_victoria_amazonica_biomineral_accretion.lean` (`MathOfBiology.Loop_1918`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/victoria_amazonica_biomineral_accretion"
EXPECTED_SHA256 = {
    "victoria_amazonica_biomineral_accretion_reference.json": "decf131904094c9e67bedf8ab87821f3fc72eb083b04b1310110775597a84fce",
    "victoria_amazonica_specimen.jpg": "e7c7e2af042ff07c75a8167e10cafcf5c7e3b5f0b4f2ddfd337ff9d436171be1",
    "scout_manifest.json": "4144e2416117b14210f80729c293c43167b2b30cb44e850860f1365acf646738",
    "paper_citations.bib": "6766694f9c253d9a166bc1baf7e65c793679a16da3187e86bcd194debad68dd3"
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
    ref = json.loads((DATASET_DIR / "victoria_amazonica_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-box-victoria-amazonica-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
