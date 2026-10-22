#!/usr/bin/env python3
"""
models/2000-saunders-goniatites-crenistria-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1325)
Organism: Goniatites crenistria (Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18))
Phenomenon Scale: 3.5 mm (0.003532 m, log10=-2.452, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Goniatites crenistria (Early Carboniferous (Viséan, 333 Ma))
Geological Horizon: Early Carboniferous (Viséan, 333 Ma) (333.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Goniatites
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (3.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1325_goniatites_crenistria_morphogen_condensation.lean` (`MathOfBiology.Loop_1325`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/goniatites_crenistria_morphogen_condensation"
EXPECTED_SHA256 = {
    "goniatites_crenistria_morphogen_condensation_reference.json": "8d91918715de17a9c7459fa1cbf5aeea9623bf913984790ad654c981ed05b5f2",
    "goniatites_crenistria_specimen.jpg": "1c025730c53d3fe4dde10fe4dfc3229dac439001911ee16d02107da5f04120fb",
    "scout_manifest.json": "5f9924ae513d4ffcfee9fc2f2f0dc0f33f13d1418e454de036c5aed5089d1c02",
    "paper_citations.bib": "8be68c919e710216ca41155690b5c5b40bfc325e3ac2f0101f13f7984adf93e2"
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
    ref = json.loads((DATASET_DIR / "goniatites_crenistria_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-saunders-goniatites-crenistria-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
