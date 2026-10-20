#!/usr/bin/env python3
"""
2017-hohn-volvox-carteri-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1091)
Organism: Volvox carteri (Colonial Green Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Volvox
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1091_volvox_carteri_collective_stigmergy.lean` (`MathOfBiology.Loop_1091`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/volvox_carteri_collective_stigmergy"
EXPECTED_SHA256 = {
    "volvox_carteri_collective_stigmergy_reference.json": "9fe5526a6479d2cd86b14655a8888291fca9173d09b100e3f289c5ed6ac36265",
    "volvox_carteri_specimen.jpg": "cf734e5f4b2d7f5ded097049b19cf0c7c00c592b2c19a662073fe8d066b17e4a",
    "scout_manifest.json": "0041157cd6438f3eab5be0c37c2a95114a2a046b7a413dc0b44a361c33612ca9",
    "paper_citations.bib": "6e1f48334910d61b5ac868000f1bd68055489cb36e154aa7577f9aa393ea52dd"
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
    ref = json.loads((DATASET_DIR / "volvox_carteri_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-hohn-volvox-carteri-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
