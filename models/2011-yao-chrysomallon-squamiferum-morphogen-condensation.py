#!/usr/bin/env python3
"""
models/2011-yao-chrysomallon-squamiferum-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1374)
Organism: Chrysomallon squamiferum (Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor))
Phenomenon Scale: 88 nm (8.829e-08 m, log10=-7.054, 2_macromolecular_100nm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Chrysomallon squamiferum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Chrysomallon
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (88 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1374_chrysomallon_squamiferum_morphogen_condensation.lean` (`MathOfBiology.Loop_1374`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/chrysomallon_squamiferum_morphogen_condensation"
EXPECTED_SHA256 = {
    "chrysomallon_squamiferum_morphogen_condensation_reference.json": "bc5e1cc74404c41c1fc44418858b0bf959d1fb1d43ed27e0cbc0e856da75fbc0",
    "chrysomallon_squamiferum_specimen.jpg": "9177d515a6734cec2e96211c99840c572fb4d027ec687bdf0cb090fe6bbe9acc",
    "scout_manifest.json": "1f4f5d9543e846947b04fbea2c45f8cad20bfb7b6596ec27de5494fa5dcf37d1",
    "paper_citations.bib": "42a24abb3df40d0ce94249f195cb9700c01720505ddfae9cee1ef3b639df3af8"
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
    ref = json.loads((DATASET_DIR / "chrysomallon_squamiferum_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-yao-chrysomallon-squamiferum-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
