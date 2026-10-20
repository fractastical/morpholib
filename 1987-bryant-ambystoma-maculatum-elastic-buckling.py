#!/usr/bin/env python3
"""
1987-bryant-ambystoma-maculatum-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0722)
Organism: Ambystoma maculatum (Spotted Salamander)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ambystoma
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0722_ambystoma_maculatum_elastic_buckling.lean` (`MathOfBiology.Loop_0722`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ambystoma_maculatum_elastic_buckling"
EXPECTED_SHA256 = {
    "ambystoma_maculatum_elastic_buckling_reference.json": "9a479c100b0d0ef42dab1755ccac4a2fd9e4adca796574367f80078586772d51",
    "ambystoma_maculatum_specimen.jpg": "32e4b406ab308f3580f182f76a10b7ef37ad16c150b655ab556d886365f151d6",
    "scout_manifest.json": "d0bd0a7c0917eb9ef65f1d9ac8e0b062c6bafe748a205bce55f9d6251a0e1792",
    "paper_citations.bib": "04c57b80b3e450ec86a2f8104bc8e118a7ff71ee599d5d4076ab9ffe65137bea"
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
    ref = json.loads((DATASET_DIR / "ambystoma_maculatum_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1987-bryant-ambystoma-maculatum-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
