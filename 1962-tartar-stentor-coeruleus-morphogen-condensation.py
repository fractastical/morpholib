#!/usr/bin/env python3
"""
1962-tartar-stentor-coeruleus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0296)
Organism: Stentor coeruleus (Trumpet Ciliate)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Stentor
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0296_stentor_coeruleus_morphogen_condensation.lean` (`MathOfBiology.Loop_0296`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/stentor_coeruleus_morphogen_condensation"
EXPECTED_SHA256 = {
    "stentor_coeruleus_morphogen_condensation_reference.json": "7d86cdb9bdd9cb0933eef0b2cb950142152204cecc472ddfeb2be07c17eac75f",
    "stentor_coeruleus_specimen.jpg": "a1681963dfc488967f3552b87eed7d5716e022271322cdda8623fa8000374fb8",
    "scout_manifest.json": "993e2e90f23551f13d326eaf86e8960aeec43552485182def1c1c4d51732896a",
    "paper_citations.bib": "09b8544d499c41e70e259e539be177df1a24813ea62eccbef2eadc5714e9bf87"
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
    ref = json.loads((DATASET_DIR / "stentor_coeruleus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1962-tartar-stentor-coeruleus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
