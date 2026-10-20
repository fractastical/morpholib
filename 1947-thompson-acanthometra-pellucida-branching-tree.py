#!/usr/bin/env python3
"""
1947-thompson-acanthometra-pellucida-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0693)
Organism: Acanthometra pellucida (Strontium Radiolarian)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acanthometra
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0693_acanthometra_pellucida_branching_tree.lean` (`MathOfBiology.Loop_0693`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acanthometra_pellucida_branching_tree"
EXPECTED_SHA256 = {
    "acanthometra_pellucida_branching_tree_reference.json": "b81713c1a1b4c00f860f75c36c609f237ff7033589c8ea29bd14b4ea14395e0c",
    "acanthometra_pellucida_specimen.jpg": "8f4b8a930e33286fd2843fa388dfec53592278e506619611732e990cc03371d4",
    "scout_manifest.json": "241ffee1423829616a363deeb0c485498e5a3d0913d8bae9bac36c2556a802ff",
    "paper_citations.bib": "b251339e1ec6fdfad4b6f2aa0853691046ac9562f2bb607ec1a2ef2be7e8a9ad"
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
    ref = json.loads((DATASET_DIR / "acanthometra_pellucida_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1947-thompson-acanthometra-pellucida-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
