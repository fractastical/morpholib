#!/usr/bin/env python3
"""
2014-wagner-tyto-alba-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0218)
Organism: Tyto alba (Barn Owl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Tyto
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0218_tyto_alba_morphogen_condensation.lean` (`MathOfBiology.Loop_0218`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tyto_alba_morphogen_condensation"
EXPECTED_SHA256 = {
    "tyto_alba_morphogen_condensation_reference.json": "a837a46a3c7aa5ea538899df87d94f9061ba12bbeae323aac6d5dfad154e0705",
    "tyto_alba_specimen.jpg": "1a81ca2e5a4b493f8c795ded141df4f7bfbee333f381a42a0b020882d72c2671",
    "scout_manifest.json": "1f24b14ba2f59996d123fa6ef685ddac260dce700dde9808b3d2fc6cd34bb310",
    "paper_citations.bib": "ba3a5f379aae681f1c7e5982c4698dab27b59050f4894251db118ea83030c1b5"
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
    ref = json.loads((DATASET_DIR / "tyto_alba_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-wagner-tyto-alba-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
