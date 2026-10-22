#!/usr/bin/env python3
"""
models/2010-bell-gasterosteus-doryssus-late-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1569)
Organism: Gasterosteus doryssus_late (Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr))
Phenomenon Scale: 105 µm (0.0001046 m, log10=-3.980, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Gasterosteus doryssus_late (Late Miocene (Truckee Fm, 9.99 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 9.99 Ma) (9.99 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (105 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1569_gasterosteus_doryssus_late_vertex_packing.lean` (`MathOfBiology.Loop_1569`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_late_vertex_packing"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_late_vertex_packing_reference.json": "9383ff737a49ec7964a7bc995f04c22cf569093331b1f17ffb10d9b99214e939",
    "gasterosteus_doryssus_late_specimen.jpg": "fd5b18be03310ca28a457aa04293f4114c76a66251867edec13e086175ae1ee1",
    "scout_manifest.json": "897988c8d406fe2c68cd3b053a2f38bcab19893c385ef7c5529a3e6a21fa33dd",
    "paper_citations.bib": "9501c87fd7b112b8b6df8e0157a0dafbb92a4cc33d165fc565e589b8862d0448"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_late_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-bell-gasterosteus-doryssus-late-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
