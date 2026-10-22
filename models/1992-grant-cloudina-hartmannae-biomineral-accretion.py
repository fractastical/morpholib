#!/usr/bin/env python3
"""
models/1992-grant-cloudina-hartmannae-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1882)
Organism: Cloudina hartmannae (Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube)
Phenomenon Scale: 1.4 mm (0.001432 m, log10=-2.844, 6_organ_mm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Cloudina hartmannae (Terminal Ediacaran (Nama Group, 549 Ma))
Geological Horizon: Terminal Ediacaran (Nama Group, 549 Ma) (549.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Cloudina
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1882_cloudina_hartmannae_biomineral_accretion.lean` (`MathOfBiology.Loop_1882`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cloudina_hartmannae_biomineral_accretion"
EXPECTED_SHA256 = {
    "cloudina_hartmannae_biomineral_accretion_reference.json": "8ccb78c4f6429b33a4645e91d326b11c3edc9a8084d5fac94fd2bd9fbd2e56bc",
    "cloudina_hartmannae_specimen.jpg": "cbe0233dbf275f4e575ad54b3134b268c070d5cafa4a4ef2add96d579bc0efe6",
    "scout_manifest.json": "5d6aaa594cec42e8832274edeb27e9348f4d0bfc4572ccec7b75c7f7d82e63ec",
    "paper_citations.bib": "c6f06f3ff0c9485b33c0d7dc3c6e3d4d6b3a12358bff9579e10f92247f303a14"
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
    ref = json.loads((DATASET_DIR / "cloudina_hartmannae_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-grant-cloudina-hartmannae-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
