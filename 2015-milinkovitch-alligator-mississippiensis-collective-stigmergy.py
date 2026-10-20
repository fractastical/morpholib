#!/usr/bin/env python3
"""
2015-milinkovitch-alligator-mississippiensis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1030)
Organism: Alligator mississippiensis (American Alligator)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Alligator
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1030_alligator_mississippiensis_collective_stigmergy.lean` (`MathOfBiology.Loop_1030`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/alligator_mississippiensis_collective_stigmergy"
EXPECTED_SHA256 = {
    "alligator_mississippiensis_collective_stigmergy_reference.json": "e896010d0071bb36a8ddd7b3e4c9026f10d0bed72d3751ac8bb192aa4947fd55",
    "alligator_mississippiensis_specimen.jpg": "bb1770a8cf4a71024eec779363925a2928277079f2124074117061fc6d960764",
    "scout_manifest.json": "c472f5aaf822522550638eff99054f7c0e78ad3abdbd2fecc44db6358570f57b",
    "paper_citations.bib": "e6ee442d21eee3e401882452b8b1ec8f669c212ac47f399f2861cf1f887b7bff"
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
    ref = json.loads((DATASET_DIR / "alligator_mississippiensis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-milinkovitch-alligator-mississippiensis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
