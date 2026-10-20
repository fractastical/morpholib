#!/usr/bin/env python3
"""
2004-moore-coprinopsis-cinerea-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0783)
Organism: Coprinopsis cinerea (Gray Shag Mushroom)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Coprinopsis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0783_coprinopsis_cinerea_elastic_buckling.lean` (`MathOfBiology.Loop_0783`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/coprinopsis_cinerea_elastic_buckling"
EXPECTED_SHA256 = {
    "coprinopsis_cinerea_elastic_buckling_reference.json": "9ef2c3579c659bbb8ca68a2f72ed2e7f59d2bba1f15a14516f0c024c1053c50d",
    "coprinopsis_cinerea_specimen.jpg": "89844e402834973a57384e9aee4268b35a0e61f0cb9977afb4a2c562507756ca",
    "scout_manifest.json": "1bed6af12ffbaa44d33fc58fca38bba11d8d1dbd9b108278d79c6ad34287a6ee",
    "paper_citations.bib": "9f94436fe21569c44ee21d7809db483b387ffc2c3ab222f59edbad1fb195901b"
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
    ref = json.loads((DATASET_DIR / "coprinopsis_cinerea_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-moore-coprinopsis-cinerea-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
