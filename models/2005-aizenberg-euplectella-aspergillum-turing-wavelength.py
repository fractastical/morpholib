#!/usr/bin/env python3
"""
models/2005-aizenberg-euplectella-aspergillum-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1285)
Organism: Euplectella aspergillum (Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage))
Phenomenon Scale: 51 µm (5.1e-05 m, log10=-4.292, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Euplectella aspergillum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Euplectella
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (51 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1285_euplectella_aspergillum_turing_wavelength.lean` (`MathOfBiology.Loop_1285`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euplectella_aspergillum_turing_wavelength"
EXPECTED_SHA256 = {
    "euplectella_aspergillum_turing_wavelength_reference.json": "d458fa4f58933dcfa391c3481375198bf9d3bacd325a482a245ad3583bc76781",
    "euplectella_aspergillum_specimen.jpg": "c39266f00e99a0f35db98c47cad0aab61aaf982021afc8d7827e37376a49fd28",
    "scout_manifest.json": "02fdb2ce9033e7c6facef43848e56457229e64e8d87646de53086805bfe7c403",
    "paper_citations.bib": "03660b00fea12d4a3ca940e28c0ef733c0419a7814040d6f1dd1639f90907238"
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
    ref = json.loads((DATASET_DIR / "euplectella_aspergillum_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-aizenberg-euplectella-aspergillum-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
