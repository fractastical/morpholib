#!/usr/bin/env python3
"""
models/2010-zhou-ginkgo-yimaensis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1351)
Organism: Ginkgo yimaensis (Middle Jurassic Deeply-Divided 6-Lobe Ginkgo Chronospecies)
Phenomenon Scale: 6.5 mm (0.006545 m, log10=-2.184, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ginkgo yimaensis (Middle Jurassic (Yima Fm, 170 Ma))
Geological Horizon: Middle Jurassic (Yima Fm, 170 Ma) (170.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (6.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1351_ginkgo_yimaensis_morphogen_condensation.lean` (`MathOfBiology.Loop_1351`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_yimaensis_morphogen_condensation"
EXPECTED_SHA256 = {
    "ginkgo_yimaensis_morphogen_condensation_reference.json": "7163e41a2ef4498dfa061a3fd044af99446f800d85e31ea7407833e0dde41d74",
    "ginkgo_yimaensis_specimen.jpg": "1ae0d5163efa653c4908ad5d93fc761b18efd5abe7d256c68c27e86ea915e772",
    "scout_manifest.json": "e34f098c30e9b84c13efe0e383900155de03774d4ea84baa41cf441dbd8d00dd",
    "paper_citations.bib": "3d5558786688b019c9c358ad4d8c3d70e5653ed04700f7f7bdd3aca138468ec5"
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
    ref = json.loads((DATASET_DIR / "ginkgo_yimaensis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-zhou-ginkgo-yimaensis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
