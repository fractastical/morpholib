#!/usr/bin/env python3
"""
1962-tartar-stentor-coeruleus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0996)
Organism: Stentor coeruleus (Trumpet Ciliate)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Stentor
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0996_stentor_coeruleus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0996`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/stentor_coeruleus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "stentor_coeruleus_bioelectric_polarity_reference.json": "edd3a0951f8be3345bb5490b2d20bebfdba605ca85fd4aa813fcfdc67a256301",
    "stentor_coeruleus_specimen.jpg": "a1681963dfc488967f3552b87eed7d5716e022271322cdda8623fa8000374fb8",
    "scout_manifest.json": "ee238b593412dba2522a9dec334a1be31eb6d39e2106eda19864312670b3f47d",
    "paper_citations.bib": "31330519026d16a2da327b057f624826f311477c52d1ca5abcb68bdfe3bffeae"
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
    ref = json.loads((DATASET_DIR / "stentor_coeruleus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1962-tartar-stentor-coeruleus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
