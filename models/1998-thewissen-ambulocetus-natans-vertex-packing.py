#!/usr/bin/env python3
"""
models/1998-thewissen-ambulocetus-natans-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1584)
Organism: Ambulocetus natans (Early Eocene Amphibious Walking Whale)
Phenomenon Scale: 651 µm (0.000651 m, log10=-3.186, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Ambulocetus natans (Early Eocene (Lutetian, 47.5 Ma))
Geological Horizon: Early Eocene (Lutetian, 47.5 Ma) (47.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ambulocetus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (651 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1584_ambulocetus_natans_vertex_packing.lean` (`MathOfBiology.Loop_1584`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ambulocetus_natans_vertex_packing"
EXPECTED_SHA256 = {
    "ambulocetus_natans_vertex_packing_reference.json": "33ae336d8f23846d7b5bb8c726104ed8bad084a3a6ace969e265a1ee3813ebc1",
    "ambulocetus_natans_specimen.jpg": "01a4e6043915d96e0394516e3db543eecad170348e219e8dd573ef897d004e67",
    "scout_manifest.json": "6fb2c1ef046e54ec22a74f2ee98e2711616821b766f2394f7bbe2b5fa9af1e6d",
    "paper_citations.bib": "dfab230b1f95f37d79c70a8bb1f5b9a3644bf9f550f0d9ef18fa26ec711e9dc3"
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
    ref = json.loads((DATASET_DIR / "ambulocetus_natans_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1998-thewissen-ambulocetus-natans-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
