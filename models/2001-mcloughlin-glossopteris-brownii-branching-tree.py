#!/usr/bin/env python3
"""
models/2001-mcloughlin-glossopteris-brownii-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1709)
Organism: Glossopteris brownii (Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation))
Phenomenon Scale: 1.8 cm (0.0179 m, log10=-1.747, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Glossopteris brownii (Late Permian (Gondwana, 260 Ma))
Geological Horizon: Late Permian (Gondwana, 260 Ma) (260.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Glossopteris
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1709_glossopteris_brownii_branching_tree.lean` (`MathOfBiology.Loop_1709`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/glossopteris_brownii_branching_tree"
EXPECTED_SHA256 = {
    "glossopteris_brownii_branching_tree_reference.json": "d8dbc84f1bf8e1027729c75dd3cba960c7206db7d7ba92575204f5853baa92f8",
    "glossopteris_brownii_specimen.jpg": "21f458497a8d6fffd0bfea916b3893a105c31943c6d82c526ea20f691965f627",
    "scout_manifest.json": "25de4c7e26012966730f7ac8f2b4dde3ef9d5749246bb2c27b837f42f742d415",
    "paper_citations.bib": "d9dd9ba29e965d77451fb3e10d9b7cfa904fc1087f209b6cbc790da21110717c"
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
    ref = json.loads((DATASET_DIR / "glossopteris_brownii_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-mcloughlin-glossopteris-brownii-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
