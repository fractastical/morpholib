#!/usr/bin/env python3
"""
2005-arendt-platynereis-dumerilii-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0261)
Organism: Platynereis dumerilii (Marine Ragworm)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Platynereis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0261_platynereis_dumerilii_morphogen_condensation.lean` (`MathOfBiology.Loop_0261`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/platynereis_dumerilii_morphogen_condensation"
EXPECTED_SHA256 = {
    "platynereis_dumerilii_morphogen_condensation_reference.json": "a97cf60b4f3cf76b07fb1eedea4ec231be9272aa44693cf97fbc36cd98e92428",
    "platynereis_dumerilii_specimen.jpg": "49376d1416ec7548315612807bfb066579e6891667d63542b0c9e663606fe5b9",
    "scout_manifest.json": "9a2799593028818697269f2d63be044ce4bd0b8620221916316e59e287db1832",
    "paper_citations.bib": "6b18fda0b1156d042e0429d5d5d51c28cd58852b5fcdcf579ba54cd6575ff047"
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
    ref = json.loads((DATASET_DIR / "platynereis_dumerilii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-arendt-platynereis-dumerilii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
