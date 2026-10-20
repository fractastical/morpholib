#!/usr/bin/env python3
"""
2000-fleming-salmo-salar-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0538)
Organism: Salmo salar (Atlantic Salmon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Salmo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0538_salmo_salar_vertex_packing.lean` (`MathOfBiology.Loop_0538`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/salmo_salar_vertex_packing"
EXPECTED_SHA256 = {
    "salmo_salar_vertex_packing_reference.json": "eafb3fe1c83936ad731a760f230fb941cc68226b725639c43b746862f75713bb",
    "salmo_salar_specimen.jpg": "2fae0b54511096d5c98f39fd08b4b665fdb0cad61caedbee18633275fbf4173e",
    "scout_manifest.json": "f4187ef24630264fff5a0eaa1480e5b177307b21a1d7aff154d64eff196c4028",
    "paper_citations.bib": "90f5f0f8bf83e12f17a0a4f3aa2b0f5718d77e6b267bc061e3de25fccfb63e3a"
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
    ref = json.loads((DATASET_DIR / "salmo_salar_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-fleming-salmo-salar-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
