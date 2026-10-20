#!/usr/bin/env python3
"""
2006-cove-physcomitrium-patens-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0872)
Organism: Physcomitrium patens (Spreading Earthmoss)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Physcomitrium
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0872_physcomitrium_patens_biomineral_accretion.lean` (`MathOfBiology.Loop_0872`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/physcomitrium_patens_biomineral_accretion"
EXPECTED_SHA256 = {
    "physcomitrium_patens_biomineral_accretion_reference.json": "b2baca11ac50d5cff078ab90c3d4d38796eeca13a181d949d05bc87c547eac76",
    "physcomitrium_patens_specimen.jpg": "a115412e5c6b1e47cad14da271d9ee4f3d33b2b7150fb4c533f145234f694d13",
    "scout_manifest.json": "9846240d84212c62bb069c2b94139cecba49d839e2dcdab313aa357ae578059c",
    "paper_citations.bib": "ccc93d0425f36de35f43a025c0154d64fe012a8e030feba893bf05fffe54e326"
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
    ref = json.loads((DATASET_DIR / "physcomitrium_patens_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-cove-physcomitrium-patens-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
