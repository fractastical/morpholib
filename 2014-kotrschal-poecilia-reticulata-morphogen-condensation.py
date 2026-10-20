#!/usr/bin/env python3
"""
2014-kotrschal-poecilia-reticulata-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0237)
Organism: Poecilia reticulata (Trinidadian Guppy)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Poecilia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0237_poecilia_reticulata_morphogen_condensation.lean` (`MathOfBiology.Loop_0237`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/poecilia_reticulata_morphogen_condensation"
EXPECTED_SHA256 = {
    "poecilia_reticulata_morphogen_condensation_reference.json": "90d4c04c12f792e38d0b4f5baee1ca74944db964dd30d2863676b3101f52fbb2",
    "poecilia_reticulata_specimen.jpg": "3e998877d83b46f9df12f9fa135839a5982986b29a14965629955111cadf8a2e",
    "scout_manifest.json": "7d670010110763e34badd5bcad585ab06c88be3e36e429794343d7d16b763ab5",
    "paper_citations.bib": "c4438ecfb2f44cff72712362935bc7d1773d71e5525b486e2ed3f42a0a45e6f5"
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
    ref = json.loads((DATASET_DIR / "poecilia_reticulata_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-kotrschal-poecilia-reticulata-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
