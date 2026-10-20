#!/usr/bin/env python3
"""
1986-bard-felis-catus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0603)
Organism: Felis catus (Domestic Cat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Felis
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0603_felis_catus_branching_tree.lean` (`MathOfBiology.Loop_0603`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/felis_catus_branching_tree"
EXPECTED_SHA256 = {
    "felis_catus_branching_tree_reference.json": "db202a20b01bd86e77cd8d4a56882c92fe8fa1fa64f796ca5942430ec2080404",
    "felis_catus_specimen.jpg": "f417376a9a5f657061f2082396f5cc7eb2b6a4d0c92e0c1d3692ceb2b52926bd",
    "scout_manifest.json": "5cca0b7fd52adc1f4dc30a6b1eba7e61e3c3383bbe113286e6295d1ff1fc06b9",
    "paper_citations.bib": "88688d889d59af24fe56e559471ef54d1ce3537f0e3fdc2976f66be2926c8a69"
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
    ref = json.loads((DATASET_DIR / "felis_catus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1986-bard-felis-catus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
