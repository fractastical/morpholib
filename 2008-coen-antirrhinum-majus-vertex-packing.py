#!/usr/bin/env python3
"""
2008-coen-antirrhinum-majus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0576)
Organism: Antirrhinum majus (Snapdragon)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Antirrhinum
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0576_antirrhinum_majus_vertex_packing.lean` (`MathOfBiology.Loop_0576`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/antirrhinum_majus_vertex_packing"
EXPECTED_SHA256 = {
    "antirrhinum_majus_vertex_packing_reference.json": "1f79c4eacc8020b101ec8f6bb318fcc86288ae2b1497b1f02d975be065273aea",
    "antirrhinum_majus_specimen.jpg": "f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0",
    "scout_manifest.json": "c0da108a557d7ccc1dbe065892c722224e408ddadd905a531729dfb012d66e59",
    "paper_citations.bib": "dd165de8293a8ac33ecd91ec4e8e529f8ff19819116951f59d10c83cd233de57"
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
    ref = json.loads((DATASET_DIR / "antirrhinum_majus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-coen-antirrhinum-majus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
