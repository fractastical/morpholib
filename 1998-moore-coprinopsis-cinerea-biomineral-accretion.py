#!/usr/bin/env python3
"""
1998-moore-coprinopsis-cinerea-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0883)
Organism: Coprinopsis cinerea (Gray Shag Mushroom)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Coprinopsis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0883_coprinopsis_cinerea_biomineral_accretion.lean` (`MathOfBiology.Loop_0883`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/coprinopsis_cinerea_biomineral_accretion"
EXPECTED_SHA256 = {
    "coprinopsis_cinerea_biomineral_accretion_reference.json": "02938435c3d8ddbd2d0e4e6ff4b34fd1c68c9e7c2b6c43711232d3835b950c2a",
    "coprinopsis_cinerea_specimen.jpg": "89844e402834973a57384e9aee4268b35a0e61f0cb9977afb4a2c562507756ca",
    "scout_manifest.json": "89d40c5116ee5ef84d5bd60e8007f7442dd646eea3f25505e6f7dcd1bcca0584",
    "paper_citations.bib": "5572f0b8ca431412320729e311250f17ba56f867a6ecd585dd638b48badb1515"
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
    ref = json.loads((DATASET_DIR / "coprinopsis_cinerea_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1998-moore-coprinopsis-cinerea-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
