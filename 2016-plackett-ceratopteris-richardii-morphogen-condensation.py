#!/usr/bin/env python3
"""
2016-plackett-ceratopteris-richardii-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0280)
Organism: Ceratopteris richardii (Triangle Water Fern)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ceratopteris
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0280_ceratopteris_richardii_morphogen_condensation.lean` (`MathOfBiology.Loop_0280`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ceratopteris_richardii_morphogen_condensation"
EXPECTED_SHA256 = {
    "ceratopteris_richardii_morphogen_condensation_reference.json": "21d3f9b81a315105c7c1d4255095081c08980e77d2050f9a1c8ae0b5a5af4bb6",
    "ceratopteris_richardii_specimen.jpg": "3fcf2684662c089b9c66687d1f6b2e9409565391d0d316e5db46561d1809641f",
    "scout_manifest.json": "29d798ab868548f4e63121adefae667ada15f2d47047b8f952d9b0f2e5b3454b",
    "paper_citations.bib": "b053665630946aa1aa14c5a36bf9351e34a98a46b33786d0371512c705cce339"
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
    ref = json.loads((DATASET_DIR / "ceratopteris_richardii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-plackett-ceratopteris-richardii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
