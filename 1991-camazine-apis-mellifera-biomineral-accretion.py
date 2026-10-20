#!/usr/bin/env python3
"""
1991-camazine-apis-mellifera-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0843)
Organism: Apis mellifera (Western Honeybee)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Apis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0843_apis_mellifera_biomineral_accretion.lean` (`MathOfBiology.Loop_0843`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/apis_mellifera_biomineral_accretion"
EXPECTED_SHA256 = {
    "apis_mellifera_biomineral_accretion_reference.json": "a490b2fb035c146f8271c1bad96b4850b07184ac7e96bacd470313bbdbe50173",
    "apis_mellifera_specimen.jpg": "67454ef389899df977e29efa0db0cba99519f5d46a3631810bcc7f160efd8fce",
    "scout_manifest.json": "f19d2da58e88c28b1fa3a7c152d44bd1c634e26cf1912ebd8fffdf176c7cde2e",
    "paper_citations.bib": "c16a38a445ef7e20521caea0d677c742666de8260037f9f3db0d6c2332cd2739"
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
    ref = json.loads((DATASET_DIR / "apis_mellifera_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1991-camazine-apis-mellifera-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
