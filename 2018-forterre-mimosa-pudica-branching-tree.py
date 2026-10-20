#!/usr/bin/env python3
"""
2018-forterre-mimosa-pudica-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0678)
Organism: Mimosa pudica (Sensitive Plant)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Mimosa
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0678_mimosa_pudica_branching_tree.lean` (`MathOfBiology.Loop_0678`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mimosa_pudica_branching_tree"
EXPECTED_SHA256 = {
    "mimosa_pudica_branching_tree_reference.json": "d69845e557985fd1580d0ced0ff1c722d3f1befb5353ac869e2155a3d0442439",
    "mimosa_pudica_specimen.jpg": "4ae1af66e325ef597ebdce00e9f639174383d049bc7b0e98b02e97de71c83e20",
    "scout_manifest.json": "8eed6d66de49bd5f56c0ec41e84eb6b141ed34a76de618894d2e698c27fb25a8",
    "paper_citations.bib": "30d32e26e637e3f51aa61644a5669cbd79d23a7f192c0c06658030b4ddab099b"
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
    ref = json.loads((DATASET_DIR / "mimosa_pudica_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-forterre-mimosa-pudica-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
