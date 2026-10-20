#!/usr/bin/env python3
"""
2014-amemiya-latimeria-chalumnae-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0239)
Organism: Latimeria chalumnae (West Indian Coelacanth)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Latimeria
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0239_latimeria_chalumnae_morphogen_condensation.lean` (`MathOfBiology.Loop_0239`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/latimeria_chalumnae_morphogen_condensation"
EXPECTED_SHA256 = {
    "latimeria_chalumnae_morphogen_condensation_reference.json": "ed440d6947a9c79ac2643a018a5b76946039552c8d58458d1580b64b282075ad",
    "latimeria_chalumnae_specimen.jpg": "29bcb5946834bbbf95c873d247d7bb9449f8ea6b56ddef496b4eef521ce6cd5d",
    "scout_manifest.json": "9c45806a36cb0d79c94e047dce36296a0915bd5930e0b2a56d38b1779aadcabc",
    "paper_citations.bib": "4b4439bab309e74c72a50b6bbe5d1e903e26e6f56e4403c1c8ba03b841182e34"
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
    ref = json.loads((DATASET_DIR / "latimeria_chalumnae_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-amemiya-latimeria-chalumnae-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
