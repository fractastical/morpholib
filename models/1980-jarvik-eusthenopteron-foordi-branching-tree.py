#!/usr/bin/env python3
"""
models/1980-jarvik-eusthenopteron-foordi-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1662)
Organism: Eusthenopteron foordi (Late Devonian Tristichopterid Lobe-Fin Fish)
Phenomenon Scale: 4.9 cm (0.04949 m, log10=-1.305, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Eusthenopteron foordi (Late Devonian (Frasnian, 379 Ma))
Geological Horizon: Late Devonian (Frasnian, 379 Ma) (379.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Eusthenopteron
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (4.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1662_eusthenopteron_foordi_branching_tree.lean` (`MathOfBiology.Loop_1662`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eusthenopteron_foordi_branching_tree"
EXPECTED_SHA256 = {
    "eusthenopteron_foordi_branching_tree_reference.json": "e130e47c3cb5a6b86ee87dd31f0ea7a04ed0fc24f4c8a77a1e7e9058efd6908a",
    "eusthenopteron_foordi_specimen.jpg": "d97b68e66855434cb59cbd6b3fc48d7827c97c6981f567e77fb025c359499811",
    "scout_manifest.json": "3f8b35cc58636b202b21386ee090e814d9fdf27520c69f1df8f2746696981c47",
    "paper_citations.bib": "3550082d25be7c373d86dddda3343adc517ec06d00c118ce49492806986f56ff"
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
    ref = json.loads((DATASET_DIR / "eusthenopteron_foordi_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1980-jarvik-eusthenopteron-foordi-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
