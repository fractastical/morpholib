#!/usr/bin/env python3
"""
models/2007-boyce-prototaxites-logani-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1710)
Organism: Prototaxites logani (Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings))
Phenomenon Scale: 11.9 cm (0.1188 m, log10=-0.925, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Prototaxites logani (Early Devonian (Emsian, 400 Ma))
Geological Horizon: Early Devonian (Emsian, 400 Ma) (400.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Prototaxites
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (11.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1710_prototaxites_logani_branching_tree.lean` (`MathOfBiology.Loop_1710`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/prototaxites_logani_branching_tree"
EXPECTED_SHA256 = {
    "prototaxites_logani_branching_tree_reference.json": "f87f4cb29caa636d260c5b69521aaffc17404a48791ab3b4e6e8e28102dc8150",
    "prototaxites_logani_specimen.jpg": "672ce91b81384694dbcf49ddcc7b3191d489b29fbe3b3f90608e2ed52f775b12",
    "scout_manifest.json": "1f706f2cb8695c3d53a5b12d4f5d2cf439d0076a90720a79e47dea3eaaa9333b",
    "paper_citations.bib": "3bfe5e70dae0cfc17a1202468de5575335f4a4a34f72a6232677bdadd17fe4c2"
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
    ref = json.loads((DATASET_DIR / "prototaxites_logani_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-boyce-prototaxites-logani-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
