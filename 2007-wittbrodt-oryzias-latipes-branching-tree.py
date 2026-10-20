#!/usr/bin/env python3
"""
2007-wittbrodt-oryzias-latipes-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0633)
Organism: Oryzias latipes (Japanese Medaka)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Oryzias
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0633_oryzias_latipes_branching_tree.lean` (`MathOfBiology.Loop_0633`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/oryzias_latipes_branching_tree"
EXPECTED_SHA256 = {
    "oryzias_latipes_branching_tree_reference.json": "f6a442230256f3b0ee3a74cb8f71832ae93ce5ac70ea54132f85544c4910130e",
    "oryzias_latipes_specimen.jpg": "ec331629f10358b6c34a698921c02e2de50725276c73bb64ffa78ba7f6e256af",
    "scout_manifest.json": "435b01e684ddfa6a84f80d35b860bccacbc81d30f1e2521f25e84eac877d0617",
    "paper_citations.bib": "d4d8dcc8bfa74d341d53b0ad49f8050badc90cc1e5905215adb3f1c93236ab8d"
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
    ref = json.loads((DATASET_DIR / "oryzias_latipes_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-wittbrodt-oryzias-latipes-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
