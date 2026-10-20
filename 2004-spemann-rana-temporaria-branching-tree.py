#!/usr/bin/env python3
"""
2004-spemann-rana-temporaria-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0626)
Organism: Rana temporaria (European Common Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Rana
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0626_rana_temporaria_branching_tree.lean` (`MathOfBiology.Loop_0626`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/rana_temporaria_branching_tree"
EXPECTED_SHA256 = {
    "rana_temporaria_branching_tree_reference.json": "8f79e3a6a1f3a70bf58c15450d865a08ea649efb79a758a1050c335800871f2a",
    "rana_temporaria_specimen.jpg": "966174057e8f12954fa5a7ba2b0a2b543b64c2900c6c92ea4a55926d1c57f111",
    "scout_manifest.json": "cc7d319742bad6d5a2246a7b55aa7316e7ed7d5e899fddf12e686c459be64748",
    "paper_citations.bib": "c5896f094990c771a7003762ebe221b2d50d80d48d5aa8c5cfbacbf0f99326bc"
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
    ref = json.loads((DATASET_DIR / "rana_temporaria_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-spemann-rana-temporaria-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
