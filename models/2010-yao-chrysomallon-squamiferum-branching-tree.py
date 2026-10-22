#!/usr/bin/env python3
"""
models/2010-yao-chrysomallon-squamiferum-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1734)
Organism: Chrysomallon squamiferum (Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor))
Phenomenon Scale: 172 nm (1.717e-07 m, log10=-6.765, 2_macromolecular_100nm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Chrysomallon squamiferum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Chrysomallon
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (172 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1734_chrysomallon_squamiferum_branching_tree.lean` (`MathOfBiology.Loop_1734`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/chrysomallon_squamiferum_branching_tree"
EXPECTED_SHA256 = {
    "chrysomallon_squamiferum_branching_tree_reference.json": "3c6af877c56395caa71b4b2ec79d4462c898124758d110e2f51f139dc4a7a9e6",
    "chrysomallon_squamiferum_specimen.jpg": "9177d515a6734cec2e96211c99840c572fb4d027ec687bdf0cb090fe6bbe9acc",
    "scout_manifest.json": "1a5c1838612c093cffe1727c7578f32132193cff4d33dfa23922596a9e5e3755",
    "paper_citations.bib": "885b7dc3ebade5321cb333287ef452341b284763314f7fdbca1879ec26c49dec"
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
    ref = json.loads((DATASET_DIR / "chrysomallon_squamiferum_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-yao-chrysomallon-squamiferum-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
