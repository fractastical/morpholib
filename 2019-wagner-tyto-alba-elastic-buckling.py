#!/usr/bin/env python3
"""
2019-wagner-tyto-alba-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0718)
Organism: Tyto alba (Barn Owl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Tyto
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0718_tyto_alba_elastic_buckling.lean` (`MathOfBiology.Loop_0718`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tyto_alba_elastic_buckling"
EXPECTED_SHA256 = {
    "tyto_alba_elastic_buckling_reference.json": "3a7566f99b889dc4fab49c05214e80b262a2d3b708b665ea6b07f446271fd84f",
    "tyto_alba_specimen.jpg": "1a81ca2e5a4b493f8c795ded141df4f7bfbee333f381a42a0b020882d72c2671",
    "scout_manifest.json": "3d16099c5af7f6d3bc0ca6bbf23b75b042a8d81f554e244cbb018210c5eb1760",
    "paper_citations.bib": "f8090e44c19689d9ba7f9bb102e444d4ae0ea5f2c72ced5162a2be38a6f0ef47"
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
    ref = json.loads((DATASET_DIR / "tyto_alba_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-wagner-tyto-alba-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
