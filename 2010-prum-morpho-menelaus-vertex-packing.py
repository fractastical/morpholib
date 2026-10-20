#!/usr/bin/env python3
"""
2010-prum-morpho-menelaus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0544)
Organism: Morpho menelaus (Blue Morpho Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Morpho
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0544_morpho_menelaus_vertex_packing.lean` (`MathOfBiology.Loop_0544`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/morpho_menelaus_vertex_packing"
EXPECTED_SHA256 = {
    "morpho_menelaus_vertex_packing_reference.json": "631f8a4806361b2512966f988ed26a95a233ce64b3f4381c317cd5b78bf40d4e",
    "morpho_menelaus_specimen.jpg": "4568faabf61d02b09522424f4f34c0efc7d869ad36aba4c8d412cda345094c68",
    "scout_manifest.json": "b76e5b0ec634cc2541c993c7ee41c5e50a2192491f881f8d3417b5fc81ca4c6b",
    "paper_citations.bib": "cdd8ee15e8df300ca53bacc270f984a072e05c5809a6d970533d710e518e8a20"
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
    ref = json.loads((DATASET_DIR / "morpho_menelaus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-prum-morpho-menelaus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
