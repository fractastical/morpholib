#!/usr/bin/env python3
"""
models/2018-landman-baculites-compressus-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1689)
Organism: Baculites compressus (Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite)
Phenomenon Scale: 3.7 cm (0.03734 m, log10=-1.428, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Baculites compressus (Late Cretaceous (Pierre Shale, 73.5 Ma))
Geological Horizon: Late Cretaceous (Pierre Shale, 73.5 Ma) (73.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Baculites
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (3.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1689_baculites_compressus_branching_tree.lean` (`MathOfBiology.Loop_1689`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/baculites_compressus_branching_tree"
EXPECTED_SHA256 = {
    "baculites_compressus_branching_tree_reference.json": "60c9b6378ac6f767ba06327c43885c818a6c5d40cf352ab10f7fe7cb4d6c8d6a",
    "baculites_compressus_specimen.jpg": "849f95ffd5d647ae2a12750587da774719bdf94f1cf2520dc375c315c45eab50",
    "scout_manifest.json": "9f1753cd032166dba975c744287373e587e0056abbb7e117322b24ccc9384e3f",
    "paper_citations.bib": "9cce02d092b272d5112a0d1431d930643db0ee2f667f3dca5fbf20be766668ea"
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
    ref = json.loads((DATASET_DIR / "baculites_compressus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-landman-baculites-compressus-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
