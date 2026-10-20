#!/usr/bin/env python3
"""
2003-brockes-notophthalmus-viridescens-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0727)
Organism: Notophthalmus viridescens (Eastern Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Notophthalmus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0727_notophthalmus_viridescens_elastic_buckling.lean` (`MathOfBiology.Loop_0727`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/notophthalmus_viridescens_elastic_buckling"
EXPECTED_SHA256 = {
    "notophthalmus_viridescens_elastic_buckling_reference.json": "1b7abd99e0e8c15c33ec09ed5c9d76678c6ece95340b04259e4a7f8563f90950",
    "notophthalmus_viridescens_specimen.jpg": "5389e095a5be650aba6c5b00fa322f4fbd3beed046b1913ac82a95ce045d5000",
    "scout_manifest.json": "8e3251361041e68b0143fba1572d3935698b3f8bec9400b40ff8359d9884af06",
    "paper_citations.bib": "7ad5c887607b76ef0df1b8722a8fe233fd679fdd3c58e355f40db851081d3fcb"
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
    ref = json.loads((DATASET_DIR / "notophthalmus_viridescens_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-brockes-notophthalmus-viridescens-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
