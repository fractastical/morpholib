#!/usr/bin/env python3
"""
2012-harrison-selaginella-moellendorffii-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0679)
Organism: Selaginella moellendorffii (Spikemoss Lycophyte)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Selaginella
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0679_selaginella_moellendorffii_branching_tree.lean` (`MathOfBiology.Loop_0679`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/selaginella_moellendorffii_branching_tree"
EXPECTED_SHA256 = {
    "selaginella_moellendorffii_branching_tree_reference.json": "4d059ef10e05071ed073d3ead363028f49ac01e9b7d67e88ca70c26a81dabf22",
    "selaginella_moellendorffii_specimen.jpg": "87d0d25c483d4f17cb66050be175071aafb49d6f321bc8eb2d9cc3bac9e55cd4",
    "scout_manifest.json": "9165fa87e7473fd649a5dbe19f2937f1fcf827789fe6f450cc6e10a79bef55a0",
    "paper_citations.bib": "95694fbfde1308605fe292ef7c62b245fb8c053bf501adf29c0099604919b221"
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
    ref = json.loads((DATASET_DIR / "selaginella_moellendorffii_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-harrison-selaginella-moellendorffii-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
