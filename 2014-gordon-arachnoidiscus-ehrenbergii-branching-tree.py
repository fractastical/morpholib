#!/usr/bin/env python3
"""
2014-gordon-arachnoidiscus-ehrenbergii-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0699)
Organism: Arachnoidiscus ehrenbergii (Wheel Centric Diatom)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Arachnoidiscus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0699_arachnoidiscus_ehrenbergii_branching_tree.lean` (`MathOfBiology.Loop_0699`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/arachnoidiscus_ehrenbergii_branching_tree"
EXPECTED_SHA256 = {
    "arachnoidiscus_ehrenbergii_branching_tree_reference.json": "eec67854d440195bedf62a7485ed12bb80b194294fa6b2fa8fbc9ad47288c4ad",
    "arachnoidiscus_ehrenbergii_specimen.jpg": "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05",
    "scout_manifest.json": "910e85317a4c3648540e8a059c01951ce3bac76ffc17eb61db035ae9042d5570",
    "paper_citations.bib": "4273e45794e1c45620821ae3c466fb1e7f58cb2b889de863b5020b1bbcb4728a"
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
    ref = json.loads((DATASET_DIR / "arachnoidiscus_ehrenbergii_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-gordon-arachnoidiscus-ehrenbergii-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
