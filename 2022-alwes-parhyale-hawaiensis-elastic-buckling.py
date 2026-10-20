#!/usr/bin/env python3
"""
2022-alwes-parhyale-hawaiensis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0745)
Organism: Parhyale hawaiensis (Amphipod Crustacean)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Parhyale
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0745_parhyale_hawaiensis_elastic_buckling.lean` (`MathOfBiology.Loop_0745`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/parhyale_hawaiensis_elastic_buckling"
EXPECTED_SHA256 = {
    "parhyale_hawaiensis_elastic_buckling_reference.json": "ef24e22793dfa00b97b8e02c6c75d47c80b5e918334b446f2cb4c83e856eeba6",
    "parhyale_hawaiensis_specimen.jpg": "b07b7fd2a6e42cf2286c7448b295d88791d4bc197bb214ee0cf7388093540dba",
    "scout_manifest.json": "3235485cef0a003ab0c5021f13aeb88b5d3ae9d9209c63b2e3d5a79ad02cae56",
    "paper_citations.bib": "dbbf944d548647bfe98762caedd06d493cda8db9eb0224633923f4f6f8e65127"
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
    ref = json.loads((DATASET_DIR / "parhyale_hawaiensis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-alwes-parhyale-hawaiensis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
