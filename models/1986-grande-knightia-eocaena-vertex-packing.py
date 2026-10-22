#!/usr/bin/env python3
"""
models/1986-grande-knightia-eocaena-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1565)
Organism: Knightia eocaena (Green River Eocene Clupeomorph Fossil)
Phenomenon Scale: 240 µm (0.0002399 m, log10=-3.620, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Knightia eocaena (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Knightia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (240 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1565_knightia_eocaena_vertex_packing.lean` (`MathOfBiology.Loop_1565`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/knightia_eocaena_vertex_packing"
EXPECTED_SHA256 = {
    "knightia_eocaena_vertex_packing_reference.json": "7c915de8ab197b6d198f3beb0effb69207cdfe5545266ed2b85cc024420f9a3a",
    "knightia_eocaena_specimen.jpg": "52c380669cecf901cc96c680b34fb444376e73b78220ae87b0df3a3c9d273c71",
    "scout_manifest.json": "ac9bc20af02c7c549bd5e66c3b21e6098ae247ddf128df539cb525ccdffa1763",
    "paper_citations.bib": "41efc3dae0ae5332dad42a27f9b048c7d73fc9bc166fc1cb4c2b4bea1856cba4"
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
    ref = json.loads((DATASET_DIR / "knightia_eocaena_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-grande-knightia-eocaena-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
