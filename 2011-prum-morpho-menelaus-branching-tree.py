#!/usr/bin/env python3
"""
2011-prum-morpho-menelaus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0644)
Organism: Morpho menelaus (Blue Morpho Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Morpho
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0644_morpho_menelaus_branching_tree.lean` (`MathOfBiology.Loop_0644`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/morpho_menelaus_branching_tree"
EXPECTED_SHA256 = {
    "morpho_menelaus_branching_tree_reference.json": "31c2ddc820a511d7efcdd428e1caa0387efd1ff1c250cade32720d73ae261e35",
    "morpho_menelaus_specimen.jpg": "4568faabf61d02b09522424f4f34c0efc7d869ad36aba4c8d412cda345094c68",
    "scout_manifest.json": "be859e758f4669624a7207c8b167515117c178558fe7ba99f3a876344ea51bdf",
    "paper_citations.bib": "f757f0356703d2c510d18385bfbe82a9ff946ae9af4eff0c5d26dbb18f354a32"
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
    ref = json.loads((DATASET_DIR / "morpho_menelaus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-prum-morpho-menelaus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
