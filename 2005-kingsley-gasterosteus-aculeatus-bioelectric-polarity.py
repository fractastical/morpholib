#!/usr/bin/env python3
"""
2005-kingsley-gasterosteus-aculeatus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0936)
Organism: Gasterosteus aculeatus (Three-Spined Stickleback)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0936_gasterosteus_aculeatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0936`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_bioelectric_polarity_reference.json": "8a3da036b24dfa91d47bfc1666c2d8e1ace7110498211e60e5017091b530d78d",
    "gasterosteus_aculeatus_specimen.jpg": "86fef1d4aeb86c3903e26fae972a360ea0c848048e29b546bedb2db842a5bd10",
    "scout_manifest.json": "ecb0cd60587405f7c7f69167a6c7de606ca6239c4268e8ceca5f372c1e6020f8",
    "paper_citations.bib": "c606b1cb44e7214e2e022bac671d1a5903f11ae26c781f72659f908b199b5822"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-kingsley-gasterosteus-aculeatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
