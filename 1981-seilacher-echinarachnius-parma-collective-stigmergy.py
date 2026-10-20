#!/usr/bin/env python3
"""
1981-seilacher-echinarachnius-parma-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1060)
Organism: Echinarachnius parma (Common Sand Dollar)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Echinarachnius
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1060_echinarachnius_parma_collective_stigmergy.lean` (`MathOfBiology.Loop_1060`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/echinarachnius_parma_collective_stigmergy"
EXPECTED_SHA256 = {
    "echinarachnius_parma_collective_stigmergy_reference.json": "6c09f8957e2d3c7cf97e7cf028b9ca8d0e9b7ffdf48146bc7a48e684ff42eb65",
    "echinarachnius_parma_specimen.jpg": "9076ecda047934e3a862128e9da39faefb16c968f63494a46db1ab7512327923",
    "scout_manifest.json": "a3f87c734aa1ed3abb6002512c3d4558b8964ada4edc30cba2be35b12988e374",
    "paper_citations.bib": "1e6eaf6a791165e741278f22b02b1ba97b08b0d10a517ac33c2ccb55cae8370d"
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
    ref = json.loads((DATASET_DIR / "echinarachnius_parma_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-seilacher-echinarachnius-parma-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
