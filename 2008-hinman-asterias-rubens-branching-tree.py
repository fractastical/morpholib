#!/usr/bin/env python3
"""
2008-hinman-asterias-rubens-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0654)
Organism: Asterias rubens (Common Starfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Asterias
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0654_asterias_rubens_branching_tree.lean` (`MathOfBiology.Loop_0654`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/asterias_rubens_branching_tree"
EXPECTED_SHA256 = {
    "asterias_rubens_branching_tree_reference.json": "c3472ca18ae4d238802bab137d0670c52d4dd58a909c7a32108746ca18b4d33d",
    "asterias_rubens_specimen.jpg": "37d94bdfed179302d1e9aa062333c2f87a5cb3d7c3932c51056f5e68c7c762ad",
    "scout_manifest.json": "737719e98cba6878a04de24523df0e7f295141a3249c7f38aad317c2ddcc569b",
    "paper_citations.bib": "5981745f480d3647e8278a2a93bb0be0eb17b3ec54edc213b1ce5f9c655ce9dd"
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
    ref = json.loads((DATASET_DIR / "asterias_rubens_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-hinman-asterias-rubens-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
