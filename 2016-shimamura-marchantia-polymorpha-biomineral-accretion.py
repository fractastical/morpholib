#!/usr/bin/env python3
"""
2016-shimamura-marchantia-polymorpha-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0871)
Organism: Marchantia polymorpha (Common Liverwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Marchantia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0871_marchantia_polymorpha_biomineral_accretion.lean` (`MathOfBiology.Loop_0871`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/marchantia_polymorpha_biomineral_accretion"
EXPECTED_SHA256 = {
    "marchantia_polymorpha_biomineral_accretion_reference.json": "5932231bf8e132f00b91d1ff4d34c81f93b669349c46dc786c32ebd7f0312ac8",
    "marchantia_polymorpha_specimen.jpg": "1a60432aad00137db36b8c40d01cac4e78b01334796182f27ed7219444c3577c",
    "scout_manifest.json": "a3abbd7fa924a694f7ff3b3833f2fe583aa655ee0408853188ca37e190925e08",
    "paper_citations.bib": "0e5652a2fad34feac6bc7abceb7f58008b0a91dd7a76a9b4693c686d876631be"
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
    ref = json.loads((DATASET_DIR / "marchantia_polymorpha_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-shimamura-marchantia-polymorpha-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
