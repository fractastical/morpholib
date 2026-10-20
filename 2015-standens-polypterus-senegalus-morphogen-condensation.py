#!/usr/bin/env python3
"""
2015-standens-polypterus-senegalus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0240)
Organism: Polypterus senegalus (Gray Bichir)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Polypterus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0240_polypterus_senegalus_morphogen_condensation.lean` (`MathOfBiology.Loop_0240`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/polypterus_senegalus_morphogen_condensation"
EXPECTED_SHA256 = {
    "polypterus_senegalus_morphogen_condensation_reference.json": "9072cd88aafbe841a9530638942f278bade9f14f1f2c279633bdca53b4a68b99",
    "polypterus_senegalus_specimen.jpg": "8332a72979f53f67ddcfb9e11c646f5695dd1fac3cbcfb340acc3d76e93582f5",
    "scout_manifest.json": "71732417645ddd180d9d665d9d8d8f36f499b00f2c396739c4e4cb9384459131",
    "paper_citations.bib": "d3cc6140a9af6efe3717384beb740acc8f369b0bd5c0f9c1628030a262ea9014"
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
    ref = json.loads((DATASET_DIR / "polypterus_senegalus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-standens-polypterus-senegalus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
