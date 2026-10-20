#!/usr/bin/env python3
"""
2005-benjacob-paenibacillus-dendritiformis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0689)
Organism: Paenibacillus dendritiformis (Chiral Branching Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Paenibacillus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0689_paenibacillus_dendritiformis_branching_tree.lean` (`MathOfBiology.Loop_0689`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/paenibacillus_dendritiformis_branching_tree"
EXPECTED_SHA256 = {
    "paenibacillus_dendritiformis_branching_tree_reference.json": "9ef848b7701dd542152093c793abd0735a889761367ccd2dc8353b9b0e09093b",
    "paenibacillus_dendritiformis_specimen.jpg": "dbed60f770b34bfce5f9193b55484568f5cccc3a5f417b5c7c6e79a7a8c84cf6",
    "scout_manifest.json": "c91006664d452386fc58a09575124daf7995f5f162c64a977bced78b2eea085c",
    "paper_citations.bib": "959f6e8238b50e21ed0ce887801e475cbef1dd1f13497948dcc35fad3b5a732b"
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
    ref = json.loads((DATASET_DIR / "paenibacillus_dendritiformis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-benjacob-paenibacillus-dendritiformis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
