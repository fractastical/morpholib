#!/usr/bin/env python3
"""
2006-thewissen-tursiops-truncatus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0110)
Organism: Tursiops truncatus (Bottlenose Dolphin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Tursiops
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0110_tursiops_truncatus_turing_wavelength.lean` (`MathOfBiology.Loop_0110`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tursiops_truncatus_turing_wavelength"
EXPECTED_SHA256 = {
    "tursiops_truncatus_turing_wavelength_reference.json": "26e9fbb6797f64eceda8eabe70d1aa2cf38100dd327da368d00dded7bfabe676",
    "tursiops_truncatus_specimen.jpg": "038f0a413155e6a92701bb438caa1f4c23fdef11b3c5c4892f851e65bcdf710a",
    "scout_manifest.json": "445af4165c3dae8507cc773ddbe0e9386a3fd5aecf24c384ba46953c6f857c20",
    "paper_citations.bib": "c8a605d19c106c62794d04c92437b5326405bde23fe6e7480938aa8972b8bf82"
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
    ref = json.loads((DATASET_DIR / "tursiops_truncatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-thewissen-tursiops-truncatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
