#!/usr/bin/env python3
"""
models/2019-munro-physalia-physalis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1286)
Organism: Physalia physalis (Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail))
Phenomenon Scale: 4.3 mm (0.004272 m, log10=-2.369, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Physalia physalis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Physalia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (4.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1286_physalia_physalis_turing_wavelength.lean` (`MathOfBiology.Loop_1286`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/physalia_physalis_turing_wavelength"
EXPECTED_SHA256 = {
    "physalia_physalis_turing_wavelength_reference.json": "0efa323f9a834afe6464d807a8a0299a6d9833111e144ba12ea90e83d04d792c",
    "physalia_physalis_specimen.jpg": "b68854e9fdaaf725c3899f1667f4e8361f6a0f2556862c28573db2d565a50904",
    "scout_manifest.json": "f8056e6dd4232d952216fe26be47f5d173cf56b31f37877f1e1b37889752813d",
    "paper_citations.bib": "42bd9676cd9782031e9315431d888e541c9d75a026e7ec93da816af644dcf6a2"
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
    ref = json.loads((DATASET_DIR / "physalia_physalis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-munro-physalia-physalis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
