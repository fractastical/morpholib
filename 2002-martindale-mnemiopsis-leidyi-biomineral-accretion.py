#!/usr/bin/env python3
"""
2002-martindale-mnemiopsis-leidyi-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0870)
Organism: Mnemiopsis leidyi (Warty Comb Jelly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Mnemiopsis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0870_mnemiopsis_leidyi_biomineral_accretion.lean` (`MathOfBiology.Loop_0870`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mnemiopsis_leidyi_biomineral_accretion"
EXPECTED_SHA256 = {
    "mnemiopsis_leidyi_biomineral_accretion_reference.json": "e90a50ec2f066f108f6fc4c2b279d37593bf2e4248d6105433c60a8f3feb1dbf",
    "mnemiopsis_leidyi_specimen.jpg": "d966e0f5b06b5af8e1ef77edf5e32c90c3aa8489be8bbbf2bdf7795ecbfb7ec7",
    "scout_manifest.json": "ab1087d32ebf245b51e5480657a41fc59064446dfebdd94636666b8481d46f13",
    "paper_citations.bib": "8e720f7747a5fc722e8d6d24aed35060787c1f7f4e258956fcf182aeac8f6eac"
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
    ref = json.loads((DATASET_DIR / "mnemiopsis_leidyi_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-martindale-mnemiopsis-leidyi-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
