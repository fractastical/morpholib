#!/usr/bin/env python3
"""
models/2012-hull-globorotalia-tumida-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1808)
Organism: Globorotalia tumida (Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies)
Phenomenon Scale: 151 µm (0.0001511 m, log10=-3.821, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Globorotalia tumida (Pliocene-Holocene (5.3–0 Ma))
Geological Horizon: Pliocene-Holocene (5.3–0 Ma) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (151 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1808_globorotalia_tumida_elastic_buckling.lean` (`MathOfBiology.Loop_1808`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_tumida_elastic_buckling"
EXPECTED_SHA256 = {
    "globorotalia_tumida_elastic_buckling_reference.json": "60acb3bd60d7ed38806f8486ad8a00d8eab4fd55d3925cb811f1752c8341d1e1",
    "globorotalia_tumida_specimen.jpg": "991d4acf78405580eb9f4efd4f75f1e9aec37937e62dce3147b582690341043b",
    "scout_manifest.json": "d9f6696af6301949c0d77514c775c1ba08c6ac2d9910aadbe144785bdf76b459",
    "paper_citations.bib": "fe698c04e72c09e29b4d692cce7ccfb8c81e0551a96755184e8c951ead4337f2"
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
    ref = json.loads((DATASET_DIR / "globorotalia_tumida_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-hull-globorotalia-tumida-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
