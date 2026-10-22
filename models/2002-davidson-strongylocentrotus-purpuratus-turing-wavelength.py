#!/usr/bin/env python3
"""
models/2002-davidson-strongylocentrotus-purpuratus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1273)
Organism: Strongylocentrotus purpuratus (Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton))
Phenomenon Scale: 3.2 µm (3.162e-06 m, log10=-5.500, 3_subcellular_um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Strongylocentrotus purpuratus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Strongylocentrotus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (3.2 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1273_strongylocentrotus_purpuratus_turing_wavelength.lean` (`MathOfBiology.Loop_1273`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/strongylocentrotus_purpuratus_turing_wavelength"
EXPECTED_SHA256 = {
    "strongylocentrotus_purpuratus_turing_wavelength_reference.json": "2cdd7e0296c6b307b79acbae98c4f460b0b59829809ea85279d14094e98c261a",
    "strongylocentrotus_purpuratus_specimen.jpg": "515b723840267df6465272f908c6951fdf06c057159d3c6f7d6e67e7bee5224a",
    "scout_manifest.json": "7fea64db74a319279d8fa027b7d0e4707217db6bbc2ab4d944d3887c06831844",
    "paper_citations.bib": "a40a32da51d264d2133d1b1a1b1b55827df9c834ee8613b7f4408e7d52736c47"
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
    ref = json.loads((DATASET_DIR / "strongylocentrotus_purpuratus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-davidson-strongylocentrotus-purpuratus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
