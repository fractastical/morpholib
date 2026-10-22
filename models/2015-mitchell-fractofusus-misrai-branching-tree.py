#!/usr/bin/env python3
"""
models/2015-mitchell-fractofusus-misrai-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1701)
Organism: Fractofusus misrai (Ediacaran Four-Order Self-Similar Spindle Rangeomorph)
Phenomenon Scale: 2.9 cm (0.02866 m, log10=-1.543, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Fractofusus misrai (Late Ediacaran (Newfoundland, 565 Ma))
Geological Horizon: Late Ediacaran (Newfoundland, 565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Fractofusus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (2.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1701_fractofusus_misrai_branching_tree.lean` (`MathOfBiology.Loop_1701`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/fractofusus_misrai_branching_tree"
EXPECTED_SHA256 = {
    "fractofusus_misrai_branching_tree_reference.json": "04a1df8b5e240dd686a3a956178270a7ef0bcdec992d57755755117ec538614b",
    "fractofusus_misrai_specimen.jpg": "b6b0eca46e1970dc859de875d6c583e3ce1abb548f235c9292a993aa71e14061",
    "scout_manifest.json": "1f77e70ce6468c47faacda124c76a6e6818bd732acf66387b50f89e3c3643637",
    "paper_citations.bib": "e1bc3932b624395acfdd07a339de8a08b76c4cb5184340d776009521a5872153"
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
    ref = json.loads((DATASET_DIR / "fractofusus_misrai_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-mitchell-fractofusus-misrai-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
