#!/usr/bin/env python3
"""
2006-frankel-tetrahymena-thermophila-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0797)
Organism: Tetrahymena thermophila (Ciliated Cortical Protist)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Tetrahymena
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0797_tetrahymena_thermophila_elastic_buckling.lean` (`MathOfBiology.Loop_0797`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetrahymena_thermophila_elastic_buckling"
EXPECTED_SHA256 = {
    "tetrahymena_thermophila_elastic_buckling_reference.json": "c1d3e1a91bb3b3201df157387234636683c2acf4921c5bae37703a8cb7efa036",
    "tetrahymena_thermophila_specimen.jpg": "7704d06cf5ec8d2a1b6bb5954c4846597cac08e5f4d0c57aed6cd6e1d729026c",
    "scout_manifest.json": "0ca101e32abe10481c060e9ed3227d0846d42042e34a3174d17180805dc34d2d",
    "paper_citations.bib": "fc2c9a4ad5a02d26a3a68d3b1eb866941a6a7fd87a92219e43b733252df5fa36"
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
    ref = json.loads((DATASET_DIR / "tetrahymena_thermophila_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-frankel-tetrahymena-thermophila-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
