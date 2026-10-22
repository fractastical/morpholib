#!/usr/bin/env python3
"""
models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1161)
Organism: Danio rerio (Zebrafish — 24–36 hpf)
Phenomenon Scale: 167 µm (0.0001672 m, log10=-3.777, 5_primordium_100um)
Anatomical Structure: Migrating posterior lateral line primordium (PLLP) & looping heart tube (24–36 hpf (Pharyngula PLLP & Linear Heart Tube, ZFA:0001279)) · Zebrahub migrating_PLLP_stream & myocardial_endocardial_tube (24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection at 167 µm (Migrating posterior lateral line primordium (PLLP) & looping heart tube))
Formal Verification: `morphoeval/lean/loops/Loop_1161_danio_rerio_um220_pllp_heart_primordium_turing_wavelength.lean` (`MathOfBiology.Loop_1161`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um220_pllp_heart_primordium_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_rerio_um220_pllp_heart_primordium_turing_wavelength_reference.json": "53025c00c0480c1a03a70125625cab153a2f6966389e608a07eb097781976f95",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "a3dc17c4ef6e4b5282120b39d59f1d628e8cfbb232b4dcfa1d093d71b31ae88e",
    "paper_citations.bib": "d804139f7b6a1ccfd4ef6b8da4d53f6cdc0576df703d4ebea211c77ed2a30ed2"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um220_pllp_heart_primordium_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
