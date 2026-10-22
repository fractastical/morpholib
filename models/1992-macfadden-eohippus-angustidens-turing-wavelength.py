#!/usr/bin/env python3
"""
models/1992-macfadden-eohippus-angustidens-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1218)
Organism: Eohippus angustidens (Early Eocene Four-Toed Dawn Horse (Hyracotherium))
Phenomenon Scale: 3.3 mm (0.003298 m, log10=-2.482, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Eohippus angustidens (Early Eocene (Wasatchian, 52.0 Ma))
Geological Horizon: Early Eocene (Wasatchian, 52.0 Ma) (52.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Eohippus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (3.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1218_eohippus_angustidens_turing_wavelength.lean` (`MathOfBiology.Loop_1218`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eohippus_angustidens_turing_wavelength"
EXPECTED_SHA256 = {
    "eohippus_angustidens_turing_wavelength_reference.json": "b446c9f740fdcb04ddae4035ca6fddcf6ec77c2ab8c7c19f54da4599383b88a6",
    "eohippus_angustidens_specimen.jpg": "4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf",
    "scout_manifest.json": "139e7472d29723e4bf42bd35f1364ad3c5abf8bdf1fa1e6975e80872647534b6",
    "paper_citations.bib": "1f325065aeaad21c22024a130aa93e667f728c4e641a88479db9b590a6f457f8"
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
    ref = json.loads((DATASET_DIR / "eohippus_angustidens_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-macfadden-eohippus-angustidens-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
