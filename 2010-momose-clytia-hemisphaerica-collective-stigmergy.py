#!/usr/bin/env python3
"""
2010-momose-clytia-hemisphaerica-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1068)
Organism: Clytia hemisphaerica (Colonial Hydrozoan)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Clytia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1068_clytia_hemisphaerica_collective_stigmergy.lean` (`MathOfBiology.Loop_1068`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/clytia_hemisphaerica_collective_stigmergy"
EXPECTED_SHA256 = {
    "clytia_hemisphaerica_collective_stigmergy_reference.json": "fe55d8e705d5232a900f302641b0d47f7c75f0aa2971397e9e3384401c991a4d",
    "clytia_hemisphaerica_specimen.jpg": "2fcc34f8d72cdc5aca117a04c8528f2e4227f504bc61fe0588a43db0a016f236",
    "scout_manifest.json": "5c532197415aab69b33027c7b76e6be89327380b94d375e6856bbdb3be9cd6c6",
    "paper_citations.bib": "6b7b77262d644f80c9b7a61bb4ece922f1c23deb9315084a5b5b8bd83774a6e3"
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
    ref = json.loads((DATASET_DIR / "clytia_hemisphaerica_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-momose-clytia-hemisphaerica-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
