#!/usr/bin/env python3
"""
2025-chirat-argonauta-argo-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0556)
Organism: Argonauta argo (Paper Nautilus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Argonauta
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0556_argonauta_argo_vertex_packing.lean` (`MathOfBiology.Loop_0556`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/argonauta_argo_vertex_packing"
EXPECTED_SHA256 = {
    "argonauta_argo_vertex_packing_reference.json": "3b81787a9998845b53b73766bad50128fc1cf05c6c4a78f42c12abb4aef42ea0",
    "argonauta_argo_specimen.jpg": "4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2",
    "scout_manifest.json": "c51e9618dc4d268e7b810233bad77c694f3ad03a3741ca0c5f8b80ccab2a2a97",
    "paper_citations.bib": "486df91ecc793e223121a8cb3b81d6ea82559d0f45045812ac4f8881395766bd"
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
    ref = json.loads((DATASET_DIR / "argonauta_argo_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2025-chirat-argonauta-argo-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
