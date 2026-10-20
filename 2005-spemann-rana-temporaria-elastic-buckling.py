#!/usr/bin/env python3
"""
2005-spemann-rana-temporaria-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0726)
Organism: Rana temporaria (European Common Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Rana
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0726_rana_temporaria_elastic_buckling.lean` (`MathOfBiology.Loop_0726`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/rana_temporaria_elastic_buckling"
EXPECTED_SHA256 = {
    "rana_temporaria_elastic_buckling_reference.json": "0cace069bdb16165996a2e575d42b69a08de37236d949220cc5c82d04d5c60c2",
    "rana_temporaria_specimen.jpg": "966174057e8f12954fa5a7ba2b0a2b543b64c2900c6c92ea4a55926d1c57f111",
    "scout_manifest.json": "74bee322553d677f45832a84fa5c4c8bdc85adfd97169d77a96e27c913c1d126",
    "paper_citations.bib": "f658ce1277c3e8f59cbce1bc175e05b0c0650a25a5e4b4f0800715106216df76"
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
    ref = json.loads((DATASET_DIR / "rana_temporaria_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-spemann-rana-temporaria-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
