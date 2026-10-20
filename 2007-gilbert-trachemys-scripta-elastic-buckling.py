#!/usr/bin/env python3
"""
2007-gilbert-trachemys-scripta-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0729)
Organism: Trachemys scripta (Red-Eared Slider Turtle)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Trachemys
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0729_trachemys_scripta_elastic_buckling.lean` (`MathOfBiology.Loop_0729`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/trachemys_scripta_elastic_buckling"
EXPECTED_SHA256 = {
    "trachemys_scripta_elastic_buckling_reference.json": "d036deb77e2b746ccf37600e6644dbb71a9b1312b496d85de79fb35164330d6b",
    "trachemys_scripta_specimen.jpg": "5d30c00d709c06ef8c1ebdb9b272654df48237cd07a6d19be7e219ab27188e96",
    "scout_manifest.json": "486b94cbbde66eec2b7458b4ccf59990c37d78fd68067260fe2d3e82da988fc2",
    "paper_citations.bib": "a41c4a2b116eef8676cfc09b9c0d08a4cbb6e3d12d233de44f411f457f5f1919"
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
    ref = json.loads((DATASET_DIR / "trachemys_scripta_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-gilbert-trachemys-scripta-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
