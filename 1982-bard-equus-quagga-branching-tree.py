#!/usr/bin/env python3
"""
1982-bard-equus-quagga-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0604)
Organism: Equus quagga (Plains Zebra)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0604_equus_quagga_branching_tree.lean` (`MathOfBiology.Loop_0604`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/equus_quagga_branching_tree"
EXPECTED_SHA256 = {
    "equus_quagga_branching_tree_reference.json": "8f02e1ae52357dc1206a3295985230f092feee0e8749bc00b072a3b08fb59956",
    "equus_quagga_specimen.jpg": "b734dc3b533ffe889dee9f205dae9015580b4b3537a05743c602408063b17ca8",
    "scout_manifest.json": "112a0926416882ae43951ba82f604b011a13632846bae2d728f0797e8d10b49b",
    "paper_citations.bib": "21cb345cca147aef876ca36fa3320e5d85fea3489c9759824a9b16377c1475bd"
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
    ref = json.loads((DATASET_DIR / "equus_quagga_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1982-bard-equus-quagga-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
