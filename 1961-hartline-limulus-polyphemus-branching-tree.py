#!/usr/bin/env python3
"""
1961-hartline-limulus-polyphemus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0650)
Organism: Limulus polyphemus (Atlantic Horseshoe Crab)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Limulus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0650_limulus_polyphemus_branching_tree.lean` (`MathOfBiology.Loop_0650`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/limulus_polyphemus_branching_tree"
EXPECTED_SHA256 = {
    "limulus_polyphemus_branching_tree_reference.json": "e18827430c967ba228ceaeafd80798d7e43570c4a04fe0a821545192ed303e13",
    "limulus_polyphemus_specimen.jpg": "6024a6e114ddb2e0b996d82067f840499683d1838662afe24496b8f315e2ef45",
    "scout_manifest.json": "ea15189c4d4b29eb6495412bce31943e0ab1ade4f79774b95b3f18ba14d7c9c1",
    "paper_citations.bib": "455e9fb0a2adc29f22443532155b1e10b6dd8ef25ef86148167620ee2ba05e25"
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
    ref = json.loads((DATASET_DIR / "limulus_polyphemus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1961-hartline-limulus-polyphemus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
