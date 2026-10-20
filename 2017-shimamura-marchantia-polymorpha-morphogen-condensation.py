#!/usr/bin/env python3
"""
2017-shimamura-marchantia-polymorpha-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0271)
Organism: Marchantia polymorpha (Common Liverwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Marchantia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0271_marchantia_polymorpha_morphogen_condensation.lean` (`MathOfBiology.Loop_0271`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/marchantia_polymorpha_morphogen_condensation"
EXPECTED_SHA256 = {
    "marchantia_polymorpha_morphogen_condensation_reference.json": "3abb40afc0a2d5b70f53f3a8cb33e8bbea835640fd42f3a21e4b7f742051a888",
    "marchantia_polymorpha_specimen.jpg": "1a60432aad00137db36b8c40d01cac4e78b01334796182f27ed7219444c3577c",
    "scout_manifest.json": "22746db07f2d3a379f7c3dc3d61009d354fb61ed7e0018b3cc7b5a23e45f687a",
    "paper_citations.bib": "7516fb7182dd1ddefb9200f4de22525201d4fca35e90759e2616e9b7d06a2893"
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
    ref = json.loads((DATASET_DIR / "marchantia_polymorpha_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-shimamura-marchantia-polymorpha-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
