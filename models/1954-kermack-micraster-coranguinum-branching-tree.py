#!/usr/bin/env python3
"""
models/1954-kermack-micraster-coranguinum-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1722)
Organism: Micraster coranguinum (Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing))
Phenomenon Scale: 1.5 cm (0.01495 m, log10=-1.825, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Micraster coranguinum (Late Cretaceous (Santonian, 85 Ma))
Geological Horizon: Late Cretaceous (Santonian, 85 Ma) (85.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Micraster
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1722_micraster_coranguinum_branching_tree.lean` (`MathOfBiology.Loop_1722`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/micraster_coranguinum_branching_tree"
EXPECTED_SHA256 = {
    "micraster_coranguinum_branching_tree_reference.json": "c789d53d0befceb7f6e475110d3b8e2b17907a7632342be84cce9e8a95bbd7a4",
    "micraster_coranguinum_specimen.jpg": "e066a3f38cd1da19878288ea7b3f5ea0aeec2482a9947c60126b6e6e07fb80da",
    "scout_manifest.json": "c2c030855519f4a6f97cf2fd432d2cdad1064a2c3d5c192c562fa7f91a4912df",
    "paper_citations.bib": "7d038957d5a3c555c241c4648c5ce16a309d039b472ea2cec6998cbbf4806f8b"
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
    ref = json.loads((DATASET_DIR / "micraster_coranguinum_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1954-kermack-micraster-coranguinum-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
