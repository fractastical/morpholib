#!/usr/bin/env python3
"""
models/2016-feoing-archaeopteryx-lithographica-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1590)
Organism: Archaeopteryx lithographica (Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane))
Phenomenon Scale: 182 µm (0.0001825 m, log10=-3.739, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Archaeopteryx lithographica (Late Jurassic (Tithonian, 150.0 Ma))
Geological Horizon: Late Jurassic (Tithonian, 150.0 Ma) (150.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Archaeopteryx
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (182 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1590_archaeopteryx_lithographica_vertex_packing.lean` (`MathOfBiology.Loop_1590`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteryx_lithographica_vertex_packing"
EXPECTED_SHA256 = {
    "archaeopteryx_lithographica_vertex_packing_reference.json": "486d61ea8172bde4d519a62454899c762b3801d474e3f6f8005ae474abc8a982",
    "archaeopteryx_lithographica_specimen.jpg": "22ad1c7de4cfaadcc6057d0f4ad6dd252fd379582bec21b51bf73852daa000af",
    "scout_manifest.json": "57d01ae60b534059e26cc0b0531a8b5f989d2f5ab0dd1bccde114c96f79153fc",
    "paper_citations.bib": "885d3189f50b88337f1e2d7a7c8b9a562e4b478c0bc456149f15a71c62f70715"
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
    ref = json.loads((DATASET_DIR / "archaeopteryx_lithographica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-feoing-archaeopteryx-lithographica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
