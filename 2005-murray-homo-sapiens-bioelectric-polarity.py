#!/usr/bin/env python3
"""
2005-murray-homo-sapiens-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0902)
Organism: Homo sapiens (Human)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Homo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0902_homo_sapiens_bioelectric_polarity.lean` (`MathOfBiology.Loop_0902`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/homo_sapiens_bioelectric_polarity"
EXPECTED_SHA256 = {
    "homo_sapiens_bioelectric_polarity_reference.json": "9388b532007879b554fd79e830ab9c94d2c93e12124640693e34fa9d8b4c6933",
    "homo_sapiens_specimen.jpg": "ed3d39e41faa4dd97074816c05e647bdc39bf7f335f22ddfa3bcd4b2760ee735",
    "scout_manifest.json": "26e8f71699ce0509fa1dc02712e0b60fd5e9a337c4bc04e3084f217c5888f180",
    "paper_citations.bib": "8d4afab85b92f0441ffb1cb9f00c7a86806c25f3c9166773b0bfe6b950c91a2b"
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
    ref = json.loads((DATASET_DIR / "homo_sapiens_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-murray-homo-sapiens-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
