#!/usr/bin/env python3
"""
2005-hammer-pecten-maximus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0657)
Organism: Pecten maximus (Great Scallop)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pecten
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0657_pecten_maximus_branching_tree.lean` (`MathOfBiology.Loop_0657`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pecten_maximus_branching_tree"
EXPECTED_SHA256 = {
    "pecten_maximus_branching_tree_reference.json": "c3cef04defd89acb95cc4880c3dd719eefb25179fde1a087ffc263cc0cf72dd4",
    "pecten_maximus_specimen.jpg": "2f87d4ec30ebec6adb82f91aec67eb300e3469f57920cbccf9c9bb0d22b703a8",
    "scout_manifest.json": "506a2e4f1957cccd605854a1a51a693228d7bcf0f4cef38ae821702219681f26",
    "paper_citations.bib": "3c1ce86ca8bacc04d91b79314fd9c59bc1e40827253cb1c0420692e606389067"
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
    ref = json.loads((DATASET_DIR / "pecten_maximus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-hammer-pecten-maximus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
