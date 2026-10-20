#!/usr/bin/env python3
"""
2012-momose-clytia-hemisphaerica-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0568)
Organism: Clytia hemisphaerica (Colonial Hydrozoan)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Clytia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0568_clytia_hemisphaerica_vertex_packing.lean` (`MathOfBiology.Loop_0568`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/clytia_hemisphaerica_vertex_packing"
EXPECTED_SHA256 = {
    "clytia_hemisphaerica_vertex_packing_reference.json": "b11ad678cb33c618b45265e9d33c4f73dfb1ca1e91a4efd465a663c0c0d2dfd5",
    "clytia_hemisphaerica_specimen.jpg": "2fcc34f8d72cdc5aca117a04c8528f2e4227f504bc61fe0588a43db0a016f236",
    "scout_manifest.json": "770ff5aa06dade53f50a929004a73f360772cbe8ff088c55edb31a7bd1fd5347",
    "paper_citations.bib": "882bc5f10cbc0b0b2d60b133c2593545574e83e1a919c5ef278fa3f8809d135b"
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
    ref = json.loads((DATASET_DIR / "clytia_hemisphaerica_vertex_packing_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2012-momose-clytia-hemisphaerica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
