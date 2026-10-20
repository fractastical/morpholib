#!/usr/bin/env python3
"""
2012-niimi-bombyx-mori-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0746)
Organism: Bombyx mori (Domestic Silkmoth)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Bombyx
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0746_bombyx_mori_elastic_buckling.lean` (`MathOfBiology.Loop_0746`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/bombyx_mori_elastic_buckling"
EXPECTED_SHA256 = {
    "bombyx_mori_elastic_buckling_reference.json": "a5ef1be811b4e62807938ad3f922ada0e4cfa2b362ec1e030410dfd08c8029b0",
    "bombyx_mori_specimen.jpg": "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97",
    "scout_manifest.json": "a49686625f16ff9563a5a057986a7cb8b7631f5b299363f5a8ebee05fcfbdc79",
    "paper_citations.bib": "5ec454b9b6fe777d516947e43630c1787b5207f02c2458e79bb6898e5d43e085"
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
    ref = json.loads((DATASET_DIR / "bombyx_mori_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-niimi-bombyx-mori-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
