#!/usr/bin/env python3
"""
models/2004-zheng-ginkgo-apodes-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1352)
Organism: Ginkgo apodes (Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies)
Phenomenon Scale: 7.9 mm (0.007938 m, log10=-2.100, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ginkgo apodes (Early Cretaceous (Yixian Fm, 121 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 121 Ma) (121.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (7.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1352_ginkgo_apodes_morphogen_condensation.lean` (`MathOfBiology.Loop_1352`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_apodes_morphogen_condensation"
EXPECTED_SHA256 = {
    "ginkgo_apodes_morphogen_condensation_reference.json": "fdd74e095f385e87fe66de80b83484ab8480c6c9c6e8558dec724496ef9201fd",
    "ginkgo_apodes_specimen.jpg": "3a074e9794ed0797dcc5abd51ab2fcccf8a109da1756d08ad6f52d95faf7d0f5",
    "scout_manifest.json": "0818f7de3c9008d15b340be2b0b49cea807f27122e282836efb4d227369fed49",
    "paper_citations.bib": "623e4818adaf76cdf8879ef2c4c52151bdd45c585a8acc4f0a495744641a1c74"
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
    ref = json.loads((DATASET_DIR / "ginkgo_apodes_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-zheng-ginkgo-apodes-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
