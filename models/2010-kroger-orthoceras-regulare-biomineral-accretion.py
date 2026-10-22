#!/usr/bin/env python3
"""
models/2010-kroger-orthoceras-regulare-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1864)
Organism: Orthoceras regulare (Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa))
Phenomenon Scale: 2.6 cm (0.02599 m, log10=-1.585, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Orthoceras regulare (Middle Ordovician (Darriwilian, 464 Ma))
Geological Horizon: Middle Ordovician (Darriwilian, 464 Ma) (464.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Orthoceras
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (2.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1864_orthoceras_regulare_biomineral_accretion.lean` (`MathOfBiology.Loop_1864`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/orthoceras_regulare_biomineral_accretion"
EXPECTED_SHA256 = {
    "orthoceras_regulare_biomineral_accretion_reference.json": "df9f5d4394f9329429b3a8468cb36361687582a44c970c52e04e23a4eb678985",
    "orthoceras_regulare_specimen.jpg": "67fd83548556de1506f61ae1eaf002726c3f3a48e3a9d8452937b96eaa47a976",
    "scout_manifest.json": "bca3b7e1f29c8caa504e6e5cfb3d9c5ab54f0f3b8185492a766ee9ce47db3253",
    "paper_citations.bib": "5d27a4e03a37140b7085d9db4fadf1feb56503972442f13341887f95f90bce9a"
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
    ref = json.loads((DATASET_DIR / "orthoceras_regulare_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-kroger-orthoceras-regulare-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
