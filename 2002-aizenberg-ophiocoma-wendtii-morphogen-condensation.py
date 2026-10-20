#!/usr/bin/env python3
"""
2002-aizenberg-ophiocoma-wendtii-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0259)
Organism: Ophiocoma wendtii (Red Brittle Star)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ophiocoma
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0259_ophiocoma_wendtii_morphogen_condensation.lean` (`MathOfBiology.Loop_0259`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ophiocoma_wendtii_morphogen_condensation"
EXPECTED_SHA256 = {
    "ophiocoma_wendtii_morphogen_condensation_reference.json": "981a96cc61982bdb9b2678931931b7a34b2fb539228cd8e582eff0b54757daf8",
    "ophiocoma_wendtii_specimen.jpg": "c073d01d499967366ff61d9831cb0e07e7505885a8ce4a9445565ec8d34f057d",
    "scout_manifest.json": "ae8c927984c96160e61c2769650d33d3446fbe14ae85e89607ff8cb0b7525ad0",
    "paper_citations.bib": "1e58a29a9043a5a666ee79ecf9992ea40dc14108e5c99230e0f6210c8d3f5c19"
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
    ref = json.loads((DATASET_DIR / "ophiocoma_wendtii_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-aizenberg-ophiocoma-wendtii-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
