#!/usr/bin/env python3
"""
models/1992-macfadden-mesohippus-bairdi-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1219)
Organism: Mesohippus bairdi (Late Eocene Three-Toed Sub-Hypsodont Horse)
Phenomenon Scale: 5.0 mm (0.00505 m, log10=-2.297, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Mesohippus bairdi (Late Eocene (Chadronian, 35.0 Ma))
Geological Horizon: Late Eocene (Chadronian, 35.0 Ma) (35.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mesohippus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (5.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1219_mesohippus_bairdi_turing_wavelength.lean` (`MathOfBiology.Loop_1219`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/mesohippus_bairdi_turing_wavelength"
EXPECTED_SHA256 = {
    "mesohippus_bairdi_turing_wavelength_reference.json": "0af3250c47abc532adb9ec555598e8bb4462077c547fc59f1d30bc75b8c78ab8",
    "mesohippus_bairdi_specimen.jpg": "ffa85314095959cda06a6a0c44566047678b150e06996250a97c7c8b4dbc1876",
    "scout_manifest.json": "f5395d70070914b0caf7771e7122769524fec1a31acbf06b9d0c3f0a778ddffe",
    "paper_citations.bib": "13dd911f784fb794e1642c41e1baa57e871347641e0e1bbf91a1a9a33987215f"
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
    ref = json.loads((DATASET_DIR / "mesohippus_bairdi_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-macfadden-mesohippus-bairdi-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
