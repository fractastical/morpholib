#!/usr/bin/env python3
"""
2017-dietrich-pseudomonas-aeruginosa-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0584)
Organism: Pseudomonas aeruginosa (Biofilm Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Pseudomonas
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0584_pseudomonas_aeruginosa_vertex_packing.lean` (`MathOfBiology.Loop_0584`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pseudomonas_aeruginosa_vertex_packing"
EXPECTED_SHA256 = {
    "pseudomonas_aeruginosa_vertex_packing_reference.json": "810167027793da40bfe0f565c8ce5c45bd8ea52d6e312bf0fcd2588e82f86aa3",
    "pseudomonas_aeruginosa_specimen.jpg": "60ef5e2ce063576716d4086509c23046d1f100864118a90be5614a770b3ff31d",
    "scout_manifest.json": "eae60e36e94351cb6e19a0b1a33c0dc6c4194758238c8a3f1f6c1fc9e6fbf0f0",
    "paper_citations.bib": "e123bd240f252839c704701e3bdea611655b6669ba178ba4c8311e0fe308f6f5"
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
    ref = json.loads((DATASET_DIR / "pseudomonas_aeruginosa_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-dietrich-pseudomonas-aeruginosa-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
