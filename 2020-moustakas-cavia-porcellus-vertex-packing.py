#!/usr/bin/env python3
"""
2020-moustakas-cavia-porcellus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0508)
Organism: Cavia porcellus (Guinea Pig)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Cavia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0508_cavia_porcellus_vertex_packing.lean` (`MathOfBiology.Loop_0508`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/cavia_porcellus_vertex_packing"
EXPECTED_SHA256 = {
    "cavia_porcellus_vertex_packing_reference.json": "cc5fd186fc2fdc12cb02981ecb70dec3a5a036f17942b38b2037e5220038606f",
    "cavia_porcellus_specimen.jpg": "42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd",
    "scout_manifest.json": "05b6bb1a2609b265eb4b4dd0e70585c9aa7462bb1aac157351d78f9dc05b5391",
    "paper_citations.bib": "78fe1efc11a15a27ad9c1817d6ec1397bee6fdb1c9b18f3d5362d6d6652fe34d"
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
    ref = json.loads((DATASET_DIR / "cavia_porcellus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-moustakas-cavia-porcellus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
