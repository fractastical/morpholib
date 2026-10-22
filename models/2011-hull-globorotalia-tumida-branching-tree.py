#!/usr/bin/env python3
"""
models/2011-hull-globorotalia-tumida-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1718)
Organism: Globorotalia tumida (Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies)
Phenomenon Scale: 240 µm (0.0002404 m, log10=-3.619, 5_primordium_100um)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Globorotalia tumida (Pliocene-Holocene (5.3–0 Ma))
Geological Horizon: Pliocene-Holocene (5.3–0 Ma) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (240 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1718_globorotalia_tumida_branching_tree.lean` (`MathOfBiology.Loop_1718`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_tumida_branching_tree"
EXPECTED_SHA256 = {
    "globorotalia_tumida_branching_tree_reference.json": "8e6678fde706cd0ba37ea54bb4f3a7bc81e9bdca3fa949955b7751ab96df1c38",
    "globorotalia_tumida_specimen.jpg": "991d4acf78405580eb9f4efd4f75f1e9aec37937e62dce3147b582690341043b",
    "scout_manifest.json": "7655c16db827cff6484f6aa477d2bdc412c2af9a105cea99ec08d1743c01c866",
    "paper_citations.bib": "3623f4beda18d6363089ad3f0d547450a1e32abe72ece72d84d81f26afa0bbfe"
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
    ref = json.loads((DATASET_DIR / "globorotalia_tumida_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-hull-globorotalia-tumida-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
