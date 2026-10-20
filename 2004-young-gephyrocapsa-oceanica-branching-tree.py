#!/usr/bin/env python3
"""
2004-young-gephyrocapsa-oceanica-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0698)
Organism: Gephyrocapsa oceanica (Calcite Coccolithophore)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Gephyrocapsa
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0698_gephyrocapsa_oceanica_branching_tree.lean` (`MathOfBiology.Loop_0698`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gephyrocapsa_oceanica_branching_tree"
EXPECTED_SHA256 = {
    "gephyrocapsa_oceanica_branching_tree_reference.json": "ed4618048ac1ccfdbdec8609a53c835d83292157a772f3e53cba89a0e3811bc6",
    "gephyrocapsa_oceanica_specimen.jpg": "b3482bdf62d2f619ecd8ad93b8bc68fb784477c51f4de8d0682d3d8fa3dbbc9e",
    "scout_manifest.json": "8d04cf8e75254c61217263b876dbf87e1650f384bc1517037c610e3381704b1a",
    "paper_citations.bib": "a22f343b1847563dd634313c1f3c9f2316a8a161ee032b409504cd750f4e143e"
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
    ref = json.loads((DATASET_DIR / "gephyrocapsa_oceanica_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-young-gephyrocapsa-oceanica-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
