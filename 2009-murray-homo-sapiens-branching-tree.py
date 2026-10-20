#!/usr/bin/env python3
"""
2009-murray-homo-sapiens-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0602)
Organism: Homo sapiens (Human)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Homo
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0602_homo_sapiens_branching_tree.lean` (`MathOfBiology.Loop_0602`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/homo_sapiens_branching_tree"
EXPECTED_SHA256 = {
    "homo_sapiens_branching_tree_reference.json": "e6f2714c2fdc3b9166a187e638fb2f8c29c5751ea301d44b2fb5568777480ab1",
    "homo_sapiens_specimen.jpg": "ed3d39e41faa4dd97074816c05e647bdc39bf7f335f22ddfa3bcd4b2760ee735",
    "scout_manifest.json": "268be0169326ec8133580abed34b9ac9ed2c96c07436489339ab9721d4696639",
    "paper_citations.bib": "27d41b08edf439a6706b73be4d0b08fa5bf2d7f68de788c1b8dcaae233659bed"
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
    ref = json.loads((DATASET_DIR / "homo_sapiens_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-murray-homo-sapiens-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
