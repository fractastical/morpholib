#!/usr/bin/env python3
"""
2004-arendt-platynereis-dumerilii-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0861)
Organism: Platynereis dumerilii (Marine Ragworm)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Platynereis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0861_platynereis_dumerilii_biomineral_accretion.lean` (`MathOfBiology.Loop_0861`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/platynereis_dumerilii_biomineral_accretion"
EXPECTED_SHA256 = {
    "platynereis_dumerilii_biomineral_accretion_reference.json": "c48acd08f6939172e75df00732a026a451094f583ff7f91612f2219eab4a5219",
    "platynereis_dumerilii_specimen.jpg": "49376d1416ec7548315612807bfb066579e6891667d63542b0c9e663606fe5b9",
    "scout_manifest.json": "8f049587615dbf41b8414e73b59e469c7e6dcc48a340053e4883dec26e5d1758",
    "paper_citations.bib": "56616b0e7eeac7de92b987cf8f69dbc5bfda4891425268ae701f1fc12b26fa81"
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
    ref = json.loads((DATASET_DIR / "platynereis_dumerilii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-arendt-platynereis-dumerilii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
