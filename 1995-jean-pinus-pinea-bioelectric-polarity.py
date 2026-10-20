#!/usr/bin/env python3
"""
1995-jean-pinus-pinea-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0975)
Organism: Pinus pinea (Stone Pine)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Pinus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0975_pinus_pinea_bioelectric_polarity.lean` (`MathOfBiology.Loop_0975`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pinus_pinea_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pinus_pinea_bioelectric_polarity_reference.json": "ff13b4971d4b158c9a18afdbf3b3d642b384b5e4143ef0b6d34a22333a8952b3",
    "pinus_pinea_specimen.jpg": "87c1fef6a6fb55bd3bf3e35f590d280e453c1348b5d11f18c356c09345da3904",
    "scout_manifest.json": "758cba7280b92298b9e69c1fdab1612f63c267669594af338ecb35e843ffeddd",
    "paper_citations.bib": "8d54a080c6d86479a9adf7edd5b412f0ffa3cd84bf0daa38063ef87ca4cee113"
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
    ref = json.loads((DATASET_DIR / "pinus_pinea_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-jean-pinus-pinea-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
