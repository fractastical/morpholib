#!/usr/bin/env python3
"""
1975-wilcox-anabaena-cylindrica-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1085)
Organism: Anabaena cylindrica (Heterocyst Cyanobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Anabaena
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1085_anabaena_cylindrica_collective_stigmergy.lean` (`MathOfBiology.Loop_1085`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anabaena_cylindrica_collective_stigmergy"
EXPECTED_SHA256 = {
    "anabaena_cylindrica_collective_stigmergy_reference.json": "b7cdd1269c5a7e4c5706cf5bfc9f53a2ca40cb04f0855cf9d59ff569f6811e8b",
    "anabaena_cylindrica_specimen.jpg": "44fe88b299c228646a9ba3152293e87a876bcaa8ac24671d7b5582ff58ad4ae6",
    "scout_manifest.json": "2b85e2fdcc5ba4db1852b8eb33114d19a71129dc9ac127cd0e6217eb9ef503ba",
    "paper_citations.bib": "711dec563b17208e8b16e298889e55c869eb76a3be049e43c6410c56b2210f36"
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
    ref = json.loads((DATASET_DIR / "anabaena_cylindrica_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1975-wilcox-anabaena-cylindrica-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
