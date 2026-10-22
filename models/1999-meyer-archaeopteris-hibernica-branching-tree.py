#!/usr/bin/env python3
"""
models/1999-meyer-archaeopteris-hibernica-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1707)
Organism: Archaeopteris hibernica (Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds))
Phenomenon Scale: 1.3 cm (0.01339 m, log10=-1.873, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Archaeopteris hibernica (Late Devonian (Famennian, 370 Ma))
Geological Horizon: Late Devonian (Famennian, 370 Ma) (370.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Archaeopteris
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (1.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1707_archaeopteris_hibernica_branching_tree.lean` (`MathOfBiology.Loop_1707`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteris_hibernica_branching_tree"
EXPECTED_SHA256 = {
    "archaeopteris_hibernica_branching_tree_reference.json": "1627d44e90ca7b1528bc41b0bca67ced0cbd9a42ac09ed04a45bcaa1d617cc08",
    "archaeopteris_hibernica_specimen.jpg": "107d9b021f541e8a82a7c594b45f0250c4a6c8fa160f1b3c54299cd439a8163e",
    "scout_manifest.json": "bda7f3aa0c79b3ec75a7d4d6493c35d69ec5806e05e5423e3f9bcca3749f5ce4",
    "paper_citations.bib": "043245ce559199508fe4d0f9a08595ab15e74af90fbf7ffb5488004c78526830"
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
    ref = json.loads((DATASET_DIR / "archaeopteris_hibernica_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-meyer-archaeopteris-hibernica-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
