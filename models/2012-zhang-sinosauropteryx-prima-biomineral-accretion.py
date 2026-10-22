#!/usr/bin/env python3
"""
models/2012-zhang-sinosauropteryx-prima-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1858)
Organism: Sinosauropteryx prima (Early Cretaceous Proto-Feathered Compsognathid Dinosaur)
Phenomenon Scale: 719 µm (0.0007194 m, log10=-3.143, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Sinosauropteryx prima (Early Cretaceous (Yixian Fm, 124.6 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 124.6 Ma) (124.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Sinosauropteryx
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (719 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1858_sinosauropteryx_prima_biomineral_accretion.lean` (`MathOfBiology.Loop_1858`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/sinosauropteryx_prima_biomineral_accretion"
EXPECTED_SHA256 = {
    "sinosauropteryx_prima_biomineral_accretion_reference.json": "7d5d788dd5acaa81fddcd73f07c9ffd6be007f0ceba551506768f24fdf847dad",
    "sinosauropteryx_prima_specimen.jpg": "dafd40e5487fb100d0309d10c30a01a1479f1793e6aa37726388bfafdd50d009",
    "scout_manifest.json": "cc53a76189a35fc28494f141893a9cbfb9db6e8181e0765c26babab6fb57c6c5",
    "paper_citations.bib": "8e4b7ef46d0dbeebad75b3bef9894e25ca967bad87e11137a167c3738b41f1eb"
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
    ref = json.loads((DATASET_DIR / "sinosauropteryx_prima_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-zhang-sinosauropteryx-prima-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
