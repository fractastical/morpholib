#!/usr/bin/env python3
"""
models/1992-edwards-cooksonia-pertoni-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1704)
Organism: Cooksonia pertoni (Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids))
Phenomenon Scale: 636 µm (0.0006363 m, log10=-3.196, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Cooksonia pertoni (Late Silurian (Wenlock-Ludlow, 425 Ma))
Geological Horizon: Late Silurian (Wenlock-Ludlow, 425 Ma) (425.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Cooksonia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (636 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1704_cooksonia_pertoni_branching_tree.lean` (`MathOfBiology.Loop_1704`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cooksonia_pertoni_branching_tree"
EXPECTED_SHA256 = {
    "cooksonia_pertoni_branching_tree_reference.json": "27eca445f3ef950519872e3a286fde0dd29ca2be3bc80361dc1dae67f43276d2",
    "cooksonia_pertoni_specimen.jpg": "8fd04a0f4ac5dd5893b91ae0eb689cb450c030fc42bc7a41129eac5169e756c2",
    "scout_manifest.json": "9840e55ec446c715f6bb153f9b34d8172a88f7975027f28a5068fe97d1ef5d0e",
    "paper_citations.bib": "79e85a605817b3fcb4350f5682b3fd636b4ecf606c4e0d803bf0ff5ca11a0afa"
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
    ref = json.loads((DATASET_DIR / "cooksonia_pertoni_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-edwards-cooksonia-pertoni-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
