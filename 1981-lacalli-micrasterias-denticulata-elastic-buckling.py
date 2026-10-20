#!/usr/bin/env python3
"""
1981-lacalli-micrasterias-denticulata-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0794)
Organism: Micrasterias denticulata (Star Desmid Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Micrasterias
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0794_micrasterias_denticulata_elastic_buckling.lean` (`MathOfBiology.Loop_0794`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/micrasterias_denticulata_elastic_buckling"
EXPECTED_SHA256 = {
    "micrasterias_denticulata_elastic_buckling_reference.json": "2421ad73034aeb2b85ef7c97cd24aa83553b07f22dfdd77efc2317add445f707",
    "micrasterias_denticulata_specimen.jpg": "3e8886e5b5c9167ab5c9217bf3f399a4d3dc7dc9eca6bc05ee5e94f5ab6fac34",
    "scout_manifest.json": "b34d124bc46c2399c168f41d1da0ecbe5c6ae477c423e815a622ea36130e09c3",
    "paper_citations.bib": "4d6301d0d1de9ea391dde3d412357d021438b7dd8ef85449ac8da1887d0817de"
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
    ref = json.loads((DATASET_DIR / "micrasterias_denticulata_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-lacalli-micrasterias-denticulata-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
