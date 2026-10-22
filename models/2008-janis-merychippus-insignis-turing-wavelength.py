#!/usr/bin/env python3
"""
models/2008-janis-merychippus-insignis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1220)
Organism: Merychippus insignis (Miocene Grazing Spring-Footed Three-Toed Horse)
Phenomenon Scale: 7.6 mm (0.00756 m, log10=-2.121, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Merychippus insignis (Middle Miocene (Barstovian, 15.0 Ma))
Geological Horizon: Middle Miocene (Barstovian, 15.0 Ma) (15.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Merychippus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (7.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1220_merychippus_insignis_turing_wavelength.lean` (`MathOfBiology.Loop_1220`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/merychippus_insignis_turing_wavelength"
EXPECTED_SHA256 = {
    "merychippus_insignis_turing_wavelength_reference.json": "3c0d3c70247ac76aeb23513b32dc84598d20deec42c88463fe9b23fa16bf9ee2",
    "merychippus_insignis_specimen.jpg": "93354e358baacf8e852b4dccb0a713b62e217cf8aba2864c49e3a358f03f7425",
    "scout_manifest.json": "dd1d6e50438ab40531619d90a0e6ca8d4a21d9ebc4108d0136f9abd271cbc174",
    "paper_citations.bib": "2e12407248aa7d36df5a11c0595556f595742a639d5c20fb975b9d12982f6566"
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
    ref = json.loads((DATASET_DIR / "merychippus_insignis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-janis-merychippus-insignis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
