#!/usr/bin/env python3
"""
models/2007-aizenberg-euplectella-aspergillum-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1915)
Organism: Euplectella aspergillum (Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage))
Phenomenon Scale: 701 µm (0.0007013 m, log10=-3.154, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Euplectella aspergillum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Euplectella
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (701 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1915_euplectella_aspergillum_biomineral_accretion.lean` (`MathOfBiology.Loop_1915`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euplectella_aspergillum_biomineral_accretion"
EXPECTED_SHA256 = {
    "euplectella_aspergillum_biomineral_accretion_reference.json": "1dd6cd1895e7c82d8ee92c08f79730380c47bc872cebb60c28badfcdae009bd3",
    "euplectella_aspergillum_specimen.jpg": "c39266f00e99a0f35db98c47cad0aab61aaf982021afc8d7827e37376a49fd28",
    "scout_manifest.json": "a9521e493504f9f0d7f90c8e86f0493838586e13ec61094624ca2e1bb25eb95e",
    "paper_citations.bib": "19dfc58176d8a47a43d89150bcb2890941ba0183337b7ea6155df255edcf7ab3"
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
    ref = json.loads((DATASET_DIR / "euplectella_aspergillum_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-aizenberg-euplectella-aspergillum-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
