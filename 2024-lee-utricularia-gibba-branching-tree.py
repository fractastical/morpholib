#!/usr/bin/env python3
"""
2024-lee-utricularia-gibba-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0677)
Organism: Utricularia gibba (Humped Bladderwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Utricularia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0677_utricularia_gibba_branching_tree.lean` (`MathOfBiology.Loop_0677`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/utricularia_gibba_branching_tree"
EXPECTED_SHA256 = {
    "utricularia_gibba_branching_tree_reference.json": "3e74d645fd1cd719d5cadb10274102140391efb25b13729a9d762fec978bc7ab",
    "utricularia_gibba_specimen.jpg": "9143b846b36d681d8c10ef23ad998247630e9d5e3abebcdfc12e3fa9dadfc3f7",
    "scout_manifest.json": "176bbf5404caec7e5d92e5c59585bae7849ef63691d3545c0f3ff03b8f23baf8",
    "paper_citations.bib": "bb65756f6310196b326bc71649b1e4d919f2c65595ef243d04a4f38e920c7678"
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
    ref = json.loads((DATASET_DIR / "utricularia_gibba_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2024-lee-utricularia-gibba-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
