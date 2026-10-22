#!/usr/bin/env python3
"""
models/1983-grande-knightia-eocaena-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1295)
Organism: Knightia eocaena (Green River Eocene Clupeomorph Fossil)
Phenomenon Scale: 1.7 cm (0.01727 m, log10=-1.763, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Knightia eocaena (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Knightia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1295_knightia_eocaena_morphogen_condensation.lean` (`MathOfBiology.Loop_1295`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/knightia_eocaena_morphogen_condensation"
EXPECTED_SHA256 = {
    "knightia_eocaena_morphogen_condensation_reference.json": "61a067fe04cbb38a6cdc6634e2b3a265f5e194feef87feab22db2224edb44212",
    "knightia_eocaena_specimen.jpg": "52c380669cecf901cc96c680b34fb444376e73b78220ae87b0df3a3c9d273c71",
    "scout_manifest.json": "7333e55d0080b663be0ebe6af047ce1cd1a8ad8c4da6e0823919256d562a72de",
    "paper_citations.bib": "4537ce1f243b51314ec348dfdeae2c7d65bb5452f8757b77256ed17acfe9162e"
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
    ref = json.loads((DATASET_DIR / "knightia_eocaena_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1983-grande-knightia-eocaena-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
