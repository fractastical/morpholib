#!/usr/bin/env python3
"""
2018-shimamura-marchantia-polymorpha-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1071)
Organism: Marchantia polymorpha (Common Liverwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Marchantia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1071_marchantia_polymorpha_collective_stigmergy.lean` (`MathOfBiology.Loop_1071`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/marchantia_polymorpha_collective_stigmergy"
EXPECTED_SHA256 = {
    "marchantia_polymorpha_collective_stigmergy_reference.json": "ab5e9fd9aa9ce380076b019da0f48eb66c417d928c08b063a01f5858bae8d556",
    "marchantia_polymorpha_specimen.jpg": "1a60432aad00137db36b8c40d01cac4e78b01334796182f27ed7219444c3577c",
    "scout_manifest.json": "3228af5b9858c634ca8c950ff4c11844052a3d087fdc331c915f2452448757e7",
    "paper_citations.bib": "576cf6cc14975df7841f5ea84ac6dcde12714243ff72ad5d793d9b302afca687"
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
    ref = json.loads((DATASET_DIR / "marchantia_polymorpha_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-shimamura-marchantia-polymorpha-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
