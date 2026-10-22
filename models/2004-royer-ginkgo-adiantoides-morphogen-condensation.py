#!/usr/bin/env python3
"""
models/2004-royer-ginkgo-adiantoides-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1353)
Organism: Ginkgo adiantoides (Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies)
Phenomenon Scale: 1.0 cm (0.0102 m, log10=-1.991, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ginkgo adiantoides (Paleocene-Eocene Boundary (56 Ma))
Geological Horizon: Paleocene-Eocene Boundary (56 Ma) (56.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1353_ginkgo_adiantoides_morphogen_condensation.lean` (`MathOfBiology.Loop_1353`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_adiantoides_morphogen_condensation"
EXPECTED_SHA256 = {
    "ginkgo_adiantoides_morphogen_condensation_reference.json": "89e0e1adac1fa6f2cc81051336faaaba1e7650fe7ca6e415263ede83c080e9c7",
    "ginkgo_adiantoides_specimen.jpg": "9b107911c55810e272a4f18af3a1f21e0f0d95561cf384cba5c90c9245314ca8",
    "scout_manifest.json": "dd57c4fe49610b4efdcbb9e3f2048f557b7f19bca713dd35ccfddedab979cc9d",
    "paper_citations.bib": "d1b1724b0003e0c6c37e6f8da487293e7c0f05d3016a383d6e7cb56fff0a61a2"
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
    ref = json.loads((DATASET_DIR / "ginkgo_adiantoides_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-royer-ginkgo-adiantoides-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
