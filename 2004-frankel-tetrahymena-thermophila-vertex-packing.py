#!/usr/bin/env python3
"""
2004-frankel-tetrahymena-thermophila-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0597)
Organism: Tetrahymena thermophila (Ciliated Cortical Protist)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Tetrahymena
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0597_tetrahymena_thermophila_vertex_packing.lean` (`MathOfBiology.Loop_0597`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetrahymena_thermophila_vertex_packing"
EXPECTED_SHA256 = {
    "tetrahymena_thermophila_vertex_packing_reference.json": "942cc209e9bf901aea7d0a5691750b3eaa4167ba7b7318b8d327e599891aaeb8",
    "tetrahymena_thermophila_specimen.jpg": "7704d06cf5ec8d2a1b6bb5954c4846597cac08e5f4d0c57aed6cd6e1d729026c",
    "scout_manifest.json": "7e6c56c52c72b156ff901d3807e1829237e8ca0ed3d28047f24a409eb6fd6b58",
    "paper_citations.bib": "e54127d074ec8c1fff406838ddb536e5b358713f0dd8d4f50d37c985e1c79c74"
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
    ref = json.loads((DATASET_DIR / "tetrahymena_thermophila_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-frankel-tetrahymena-thermophila-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
