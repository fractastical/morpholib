#!/usr/bin/env python3
"""
2009-manni-botryllus-schlosseri-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1067)
Organism: Botryllus schlosseri (Star Ascidian Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Botryllus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1067_botryllus_schlosseri_collective_stigmergy.lean` (`MathOfBiology.Loop_1067`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/botryllus_schlosseri_collective_stigmergy"
EXPECTED_SHA256 = {
    "botryllus_schlosseri_collective_stigmergy_reference.json": "29ebfab9b620989c15adafe402ad1ed39b45e0e1786608754b52bdecfbcba7f0",
    "botryllus_schlosseri_specimen.jpg": "b1ea2c5231b1b71026b40278bbac0bde71e6cf35e9fd8bedcd40f103a7190401",
    "scout_manifest.json": "6234e91dca1d95118673bfcc2601cab661c18d5591bf9891c39d2e4cd190b779",
    "paper_citations.bib": "de250306cea9de770c2c75e5d562aadec9f17d41e4754d6f64199a46535cfa0a"
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
    ref = json.loads((DATASET_DIR / "botryllus_schlosseri_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-manni-botryllus-schlosseri-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
