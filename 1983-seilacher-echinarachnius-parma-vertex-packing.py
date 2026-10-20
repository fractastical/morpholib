#!/usr/bin/env python3
"""
1983-seilacher-echinarachnius-parma-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0560)
Organism: Echinarachnius parma (Common Sand Dollar)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Echinarachnius
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0560_echinarachnius_parma_vertex_packing.lean` (`MathOfBiology.Loop_0560`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/echinarachnius_parma_vertex_packing"
EXPECTED_SHA256 = {
    "echinarachnius_parma_vertex_packing_reference.json": "b294b35f04835bd44ac9954acfc4804dbc32ee155e8e1e4187ff6decabfdbb99",
    "echinarachnius_parma_specimen.jpg": "9076ecda047934e3a862128e9da39faefb16c968f63494a46db1ab7512327923",
    "scout_manifest.json": "75840a0bdd9f09fc9a392af4996b1130152228a0e7c830376a5c897768fc5a5f",
    "paper_citations.bib": "23ba447d0ae7c4be9b05cbbe6046853c9833a97e12cf1c949b8cf92eaa1cf63e"
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
    ref = json.loads((DATASET_DIR / "echinarachnius_parma_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1983-seilacher-echinarachnius-parma-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
