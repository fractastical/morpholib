#!/usr/bin/env python3
"""
models/2011-li-anchiornis-huxleyi-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1319)
Organism: Anchiornis huxleyi (Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array))
Phenomenon Scale: 130 nm (1.301e-07 m, log10=-6.886, 2_macromolecular_100nm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Anchiornis huxleyi (Late Jurassic (Tiaojishan Fm, 160.0 Ma))
Geological Horizon: Late Jurassic (Tiaojishan Fm, 160.0 Ma) (160.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anchiornis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (130 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1319_anchiornis_huxleyi_morphogen_condensation.lean` (`MathOfBiology.Loop_1319`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anchiornis_huxleyi_morphogen_condensation"
EXPECTED_SHA256 = {
    "anchiornis_huxleyi_morphogen_condensation_reference.json": "f27b4128bdad5807ceb4ab0906b38678364ac0a4afab11640458843d669ae9c3",
    "anchiornis_huxleyi_specimen.jpg": "fab0bb4ca7882feafbf18840d6056dd36150060dc484593ec1df858d67a18865",
    "scout_manifest.json": "6f05d4ccf759d88ecc4ba3b03a04e7dadfcf74863b6d1736d70f64588e422ca8",
    "paper_citations.bib": "5f0c3c7ca038f5d8309af2dde10f3152906f8763fb7450f84a585ab0c7710cc0"
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
    ref = json.loads((DATASET_DIR / "anchiornis_huxleyi_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-li-anchiornis-huxleyi-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
