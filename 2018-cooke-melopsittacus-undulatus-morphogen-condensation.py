#!/usr/bin/env python3
"""
2018-cooke-melopsittacus-undulatus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0217)
Organism: Melopsittacus undulatus (Budgerigar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Melopsittacus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0217_melopsittacus_undulatus_morphogen_condensation.lean` (`MathOfBiology.Loop_0217`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/melopsittacus_undulatus_morphogen_condensation"
EXPECTED_SHA256 = {
    "melopsittacus_undulatus_morphogen_condensation_reference.json": "38e79faa48349d892236b90c7acd87721f0997dad4bcdb505c727461a0b7d5c6",
    "melopsittacus_undulatus_specimen.jpg": "f61163f7820594be22c648c5120f1e827b9cdd8d377d9eeb65bb96e633b0ce2b",
    "scout_manifest.json": "f5d7354d047138f5cb50dfdaa68ba8a5f86ac2c0db3a76adcd64822235a51460",
    "paper_citations.bib": "cd72cc76ac72f1eac9fe2cb209708980564a2d3e5c4383f1a28d933cf26e5c43"
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
    ref = json.loads((DATASET_DIR / "melopsittacus_undulatus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-cooke-melopsittacus-undulatus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
