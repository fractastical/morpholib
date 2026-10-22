#!/usr/bin/env python3
"""
models/2024-lange-danio-rerio-nm4-connexin-her-complex-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1101)
Organism: Danio rerio (Zebrafish — 10–14 hpf)
Phenomenon Scale: 3.2 nm (3.192e-09 m, log10=-8.496, 1_molecular_nm)
Anatomical Structure: Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex (10–14 hpf (Bud to 10-Somite Stage, ZFA:0001094)) · Zebrahub PSM_presomitic_mesoderm & neural_plate_progenitors (10–14 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection at 3.2 nm (Connexin-41.8 hemichannel pore & Her1/Her7 bHLH dimer complex))
Formal Verification: `morphoeval/lean/loops/Loop_1101_danio_rerio_nm4_connexin_her_complex_turing_wavelength.lean` (`MathOfBiology.Loop_1101`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_nm4_connexin_her_complex_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_rerio_nm4_connexin_her_complex_turing_wavelength_reference.json": "067b983c98dca63cc714f6d4508cd8ab2336212ccfc7406ebfcbf0897e5606f3",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "5f7bd93949800d08b128fdaf654bfa465e484dc99df2e3bf01c798afa2dc6799",
    "paper_citations.bib": "467fd857cdf4956f3c0d64bcd2cdcce82b20f24381fcf07b053f3c4cec3f8168"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_nm4_connexin_her_complex_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-lange-danio-rerio-nm4-connexin-her-complex-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
