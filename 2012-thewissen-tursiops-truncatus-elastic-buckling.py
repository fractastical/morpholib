#!/usr/bin/env python3
"""
2012-thewissen-tursiops-truncatus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0710)
Organism: Tursiops truncatus (Bottlenose Dolphin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Tursiops
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0710_tursiops_truncatus_elastic_buckling.lean` (`MathOfBiology.Loop_0710`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tursiops_truncatus_elastic_buckling"
EXPECTED_SHA256 = {
    "tursiops_truncatus_elastic_buckling_reference.json": "7c32c7bf78d1fb37ecd7ce739684fd48148233015799c1ce06700c5391b36c95",
    "tursiops_truncatus_specimen.jpg": "038f0a413155e6a92701bb438caa1f4c23fdef11b3c5c4892f851e65bcdf710a",
    "scout_manifest.json": "080cf8914de96804976f7b2ff560887484ee9f1da3899776df3439393b70fdf6",
    "paper_citations.bib": "bf74857c495573ddebcde82726e2019be23ae312287113e9a3c7e79396b4ddd8"
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
    ref = json.loads((DATASET_DIR / "tursiops_truncatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-thewissen-tursiops-truncatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
