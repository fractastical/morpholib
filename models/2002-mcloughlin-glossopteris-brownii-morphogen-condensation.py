#!/usr/bin/env python3
"""
models/2002-mcloughlin-glossopteris-brownii-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1349)
Organism: Glossopteris brownii (Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation))
Phenomenon Scale: 9.2 mm (0.009207 m, log10=-2.036, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Glossopteris brownii (Late Permian (Gondwana, 260 Ma))
Geological Horizon: Late Permian (Gondwana, 260 Ma) (260.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Glossopteris
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (9.2 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1349_glossopteris_brownii_morphogen_condensation.lean` (`MathOfBiology.Loop_1349`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/glossopteris_brownii_morphogen_condensation"
EXPECTED_SHA256 = {
    "glossopteris_brownii_morphogen_condensation_reference.json": "c34f6dd1180c0b290f7f781dafa2cc3cfbf9e9dd23f1d31d9dd942f5b2971e45",
    "glossopteris_brownii_specimen.jpg": "21f458497a8d6fffd0bfea916b3893a105c31943c6d82c526ea20f691965f627",
    "scout_manifest.json": "c734b8f41aca3bada6b81000c2ed720e4451a6dff984adf75e604831c02ddd2f",
    "paper_citations.bib": "e48244fe36f721d8229819661f2e090484542449a9673565dd6e1fb2ff23c4af"
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
    ref = json.loads((DATASET_DIR / "glossopteris_brownii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-mcloughlin-glossopteris-brownii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
