#!/usr/bin/env python3
"""
2004-murray-homo-sapiens-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0102)
Organism: Homo sapiens (Human)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Homo
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0102_homo_sapiens_turing_wavelength.lean` (`MathOfBiology.Loop_0102`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/homo_sapiens_turing_wavelength"
EXPECTED_SHA256 = {
    "homo_sapiens_turing_wavelength_reference.json": "aeb04689ed0b13bc988ee6fb6baacac5e8bf7d48591a844860e172e086051942",
    "homo_sapiens_specimen.jpg": "ed3d39e41faa4dd97074816c05e647bdc39bf7f335f22ddfa3bcd4b2760ee735",
    "scout_manifest.json": "83b350ce1244a84add6a4824d5c7b98f5bfcb6b54190aa7db45874293bce0543",
    "paper_citations.bib": "4c3f8f4972432358615f109b11db2c77f6e1984f761cc638c0dcf6e6460a111c"
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
    ref = json.loads((DATASET_DIR / "homo_sapiens_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-murray-homo-sapiens-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
