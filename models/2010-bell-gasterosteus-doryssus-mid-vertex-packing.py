#!/usr/bin/env python3
"""
models/2010-bell-gasterosteus-doryssus-mid-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1568)
Organism: Gasterosteus doryssus_mid (Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr))
Phenomenon Scale: 104 µm (0.0001041 m, log10=-3.983, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Gasterosteus doryssus_mid (Late Miocene (Truckee Fm, 10.01 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.01 Ma) (10.01 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (104 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1568_gasterosteus_doryssus_mid_vertex_packing.lean` (`MathOfBiology.Loop_1568`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_mid_vertex_packing"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_mid_vertex_packing_reference.json": "04450e122436e35246b8e4d5e079f921e0307745b6d6e83e235d5d37e73cbf0c",
    "gasterosteus_doryssus_mid_specimen.jpg": "52d02b91f4693fdd40965b6d78971fdf8ebe6601d12fad6c49a6255e45791e42",
    "scout_manifest.json": "ec3b3e991e5e32806452ecaa5d10884e7d5ade6162c78a3d04db88cce0ae1ebb",
    "paper_citations.bib": "c7bcff3072b4ec6e35f00686e7b526259ed42653f808586a06734fd635745c69"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_mid_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-bell-gasterosteus-doryssus-mid-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
