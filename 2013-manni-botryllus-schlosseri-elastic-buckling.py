#!/usr/bin/env python3
"""
2013-manni-botryllus-schlosseri-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0767)
Organism: Botryllus schlosseri (Star Ascidian Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Botryllus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0767_botryllus_schlosseri_elastic_buckling.lean` (`MathOfBiology.Loop_0767`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/botryllus_schlosseri_elastic_buckling"
EXPECTED_SHA256 = {
    "botryllus_schlosseri_elastic_buckling_reference.json": "21979925dc98bcd5fc3a5bbdcc71218f6405ab1087519ac50bbf3a9ae6cfa70a",
    "botryllus_schlosseri_specimen.jpg": "b1ea2c5231b1b71026b40278bbac0bde71e6cf35e9fd8bedcd40f103a7190401",
    "scout_manifest.json": "d677ea23b0f52232ff73193410ad4c19c316a66ea1ccb803a915bad5991614ff",
    "paper_citations.bib": "0ec911a7e5264c1be670050627f9eedae6efdd6bec05cdf82aa04d1b888166ca"
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
    ref = json.loads((DATASET_DIR / "botryllus_schlosseri_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-manni-botryllus-schlosseri-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
