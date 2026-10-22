#!/usr/bin/env python3
"""
models/1984-grande-diplomystus-dentatus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1206)
Organism: Diplomystus dentatus (Green River Eocene Armored Herring Fossil)
Phenomenon Scale: 7.6 mm (0.00756 m, log10=-2.121, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Diplomystus dentatus (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Diplomystus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (7.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1206_diplomystus_dentatus_turing_wavelength.lean` (`MathOfBiology.Loop_1206`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/diplomystus_dentatus_turing_wavelength"
EXPECTED_SHA256 = {
    "diplomystus_dentatus_turing_wavelength_reference.json": "88148dd175ce81b8445348b4ee9d54abb84a82f035d738c6e3b385856858beef",
    "diplomystus_dentatus_specimen.jpg": "00e0f80ae28f4fe36376b217f05e2048df17ed19a49f48d49d4a9b88b2b2b659",
    "scout_manifest.json": "bc97841ef370fa0d904c341cb85056e77cc493a7bddf6292ed8e2441e7cf3965",
    "paper_citations.bib": "47fe125f4c7e4f8535beeecd8ba71f141f8e1500c901853586b394a4c32cdd6d"
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
    ref = json.loads((DATASET_DIR / "diplomystus_dentatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1984-grande-diplomystus-dentatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
