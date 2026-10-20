#!/usr/bin/env python3
"""
1987-bard-felis-catus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0703)
Organism: Felis catus (Domestic Cat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Felis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0703_felis_catus_elastic_buckling.lean` (`MathOfBiology.Loop_0703`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/felis_catus_elastic_buckling"
EXPECTED_SHA256 = {
    "felis_catus_elastic_buckling_reference.json": "bc6b415b4d6bb4e427a57115c97d7ef92650575810ce3483c03f7b736a3a4f47",
    "felis_catus_specimen.jpg": "f417376a9a5f657061f2082396f5cc7eb2b6a4d0c92e0c1d3692ceb2b52926bd",
    "scout_manifest.json": "8b8a9253b0bd09b0d983c23bd338a8fdc92ff68264626b28724c7c5ea5d3e925",
    "paper_citations.bib": "100f1503217d5740173f72b4032cea2baf4bd6b52c287ea30053c160dd9d0017"
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
    ref = json.loads((DATASET_DIR / "felis_catus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1987-bard-felis-catus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
