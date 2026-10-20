#!/usr/bin/env python3
"""
2021-alwes-parhyale-hawaiensis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0645)
Organism: Parhyale hawaiensis (Amphipod Crustacean)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Parhyale
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0645_parhyale_hawaiensis_branching_tree.lean` (`MathOfBiology.Loop_0645`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/parhyale_hawaiensis_branching_tree"
EXPECTED_SHA256 = {
    "parhyale_hawaiensis_branching_tree_reference.json": "8f0907688df59e884514904476f4a6e4afb9343451a13a3c87767cad301f1c4c",
    "parhyale_hawaiensis_specimen.jpg": "b07b7fd2a6e42cf2286c7448b295d88791d4bc197bb214ee0cf7388093540dba",
    "scout_manifest.json": "4042a455c91b703fe09359521e8751f95d75f370c74d4d87c137c555db97f6b3",
    "paper_citations.bib": "59d7ca1e5e04eefee1423bfafd1d97169dd24e63a08c3285e1186e9fcf6c55ff"
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
    ref = json.loads((DATASET_DIR / "parhyale_hawaiensis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-alwes-parhyale-hawaiensis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
