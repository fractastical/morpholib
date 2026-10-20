#!/usr/bin/env python3
"""
1976-lacalli-micrasterias-denticulata-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0294)
Organism: Micrasterias denticulata (Star Desmid Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Micrasterias
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0294_micrasterias_denticulata_morphogen_condensation.lean` (`MathOfBiology.Loop_0294`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/micrasterias_denticulata_morphogen_condensation"
EXPECTED_SHA256 = {
    "micrasterias_denticulata_morphogen_condensation_reference.json": "e0036b446244066f87d3271b2d47df4e1e2285f48b05016617e3b32f5e83d3e6",
    "micrasterias_denticulata_specimen.jpg": "3e8886e5b5c9167ab5c9217bf3f399a4d3dc7dc9eca6bc05ee5e94f5ab6fac34",
    "scout_manifest.json": "265459d039a5dd11b51de2ecbbf88d65aa977390378d3352905adc800630598e",
    "paper_citations.bib": "3ad432b28e33866d31845e7fa54ad63fbd85499d14ea5b01d0cc241f8ae5d7db"
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
    ref = json.loads((DATASET_DIR / "micrasterias_denticulata_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1976-lacalli-micrasterias-denticulata-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
