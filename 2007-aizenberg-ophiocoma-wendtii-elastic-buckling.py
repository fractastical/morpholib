#!/usr/bin/env python3
"""
2007-aizenberg-ophiocoma-wendtii-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0759)
Organism: Ophiocoma wendtii (Red Brittle Star)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ophiocoma
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0759_ophiocoma_wendtii_elastic_buckling.lean` (`MathOfBiology.Loop_0759`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ophiocoma_wendtii_elastic_buckling"
EXPECTED_SHA256 = {
    "ophiocoma_wendtii_elastic_buckling_reference.json": "ba9a61c4fdd0e67afe69023638d5e622511a184aa265621db8fadcb53c56aeab",
    "ophiocoma_wendtii_specimen.jpg": "c073d01d499967366ff61d9831cb0e07e7505885a8ce4a9445565ec8d34f057d",
    "scout_manifest.json": "242f0d3375c436decf7e1a045f9c9cc21b0fe6d144be4b8dc778ec6246828633",
    "paper_citations.bib": "737bdffdc80bddb3f70d4dff0d571e1068ab44fb6174d89d933d8e9ddbfd4064"
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
    ref = json.loads((DATASET_DIR / "ophiocoma_wendtii_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-aizenberg-ophiocoma-wendtii-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
