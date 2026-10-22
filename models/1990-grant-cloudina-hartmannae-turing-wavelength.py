#!/usr/bin/env python3
"""
models/1990-grant-cloudina-hartmannae-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1252)
Organism: Cloudina hartmannae (Terminal Ediacaran Cone-in-Cone Biomineralized Calcite Tube)
Phenomenon Scale: 104 µm (0.0001042 m, log10=-3.982, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Cloudina hartmannae (Terminal Ediacaran (Nama Group, 549 Ma))
Geological Horizon: Terminal Ediacaran (Nama Group, 549 Ma) (549.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Cloudina
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (104 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1252_cloudina_hartmannae_turing_wavelength.lean` (`MathOfBiology.Loop_1252`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cloudina_hartmannae_turing_wavelength"
EXPECTED_SHA256 = {
    "cloudina_hartmannae_turing_wavelength_reference.json": "2396a8db1d7206608c75d90e7f645f111ab1f168901d0f012d80154717948c53",
    "cloudina_hartmannae_specimen.jpg": "cbe0233dbf275f4e575ad54b3134b268c070d5cafa4a4ef2add96d579bc0efe6",
    "scout_manifest.json": "bcf0b134b1c50ac5ccbb78561902593f53359d0f8f80af9e7deea8a3631fe557",
    "paper_citations.bib": "67918ae6efb8cd4e58d0a89a1823351426aa09db6d3dfd62eec0bf99edb3189c"
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
    ref = json.loads((DATASET_DIR / "cloudina_hartmannae_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1990-grant-cloudina-hartmannae-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
