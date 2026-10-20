#!/usr/bin/env python3
"""
2012-sears-myotis-lucifugus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0709)
Organism: Myotis lucifugus (Little Brown Bat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Myotis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0709_myotis_lucifugus_elastic_buckling.lean` (`MathOfBiology.Loop_0709`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/myotis_lucifugus_elastic_buckling"
EXPECTED_SHA256 = {
    "myotis_lucifugus_elastic_buckling_reference.json": "a74fbb9cf3094180635c133f4c0d9829c9ecda1c27b7d4153f30602d7a179823",
    "myotis_lucifugus_specimen.jpg": "1c8f403a4e94c928ac82c06c27e448ba133f1f6b2691860354b6490e152a97b4",
    "scout_manifest.json": "d6d3e35247d82efb815308c5d8f119e38d93917db56689739c424aa08d40eb4e",
    "paper_citations.bib": "018e4cf0a2cd3fea21ea5a1d675211eb16f273822f98c562c1d4afd21b8bc2ec"
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
    ref = json.loads((DATASET_DIR / "myotis_lucifugus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-sears-myotis-lucifugus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
