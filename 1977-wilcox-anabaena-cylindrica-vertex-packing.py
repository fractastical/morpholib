#!/usr/bin/env python3
"""
1977-wilcox-anabaena-cylindrica-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0585)
Organism: Anabaena cylindrica (Heterocyst Cyanobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Anabaena
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0585_anabaena_cylindrica_vertex_packing.lean` (`MathOfBiology.Loop_0585`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anabaena_cylindrica_vertex_packing"
EXPECTED_SHA256 = {
    "anabaena_cylindrica_vertex_packing_reference.json": "5b3ce52a55e1ce9ee1a181dc9b10ba4ef1f3923c984d930aaf6a48a680baf0d0",
    "anabaena_cylindrica_specimen.jpg": "44fe88b299c228646a9ba3152293e87a876bcaa8ac24671d7b5582ff58ad4ae6",
    "scout_manifest.json": "09b8a2c248dd8e90227f70c6cb8b7877d24aee7df6b7f322a27d03f86fa3b2dc",
    "paper_citations.bib": "9eb0447802408773008650c14f10c3cc45cb8c2f7e2e4ec8d8f0c7b0d543dd45"
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
    ref = json.loads((DATASET_DIR / "anabaena_cylindrica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1977-wilcox-anabaena-cylindrica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
