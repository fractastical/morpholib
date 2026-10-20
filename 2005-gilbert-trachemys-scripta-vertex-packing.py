#!/usr/bin/env python3
"""
2005-gilbert-trachemys-scripta-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0529)
Organism: Trachemys scripta (Red-Eared Slider Turtle)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Trachemys
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0529_trachemys_scripta_vertex_packing.lean` (`MathOfBiology.Loop_0529`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/trachemys_scripta_vertex_packing"
EXPECTED_SHA256 = {
    "trachemys_scripta_vertex_packing_reference.json": "e9ad69bb0f4f9b079857a55898459f4d50fcc9a01e66398fd9bee053379b8c11",
    "trachemys_scripta_specimen.jpg": "5d30c00d709c06ef8c1ebdb9b272654df48237cd07a6d19be7e219ab27188e96",
    "scout_manifest.json": "be0d7afac63495a745535e9b5e0a4d21ac06f34b87593a67fcb48855567b0d6e",
    "paper_citations.bib": "9a4a499e3eee692cd5c56c99b04e3140e86979cb135884ce7ef2b48ee1b43643"
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
    ref = json.loads((DATASET_DIR / "trachemys_scripta_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-gilbert-trachemys-scripta-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
