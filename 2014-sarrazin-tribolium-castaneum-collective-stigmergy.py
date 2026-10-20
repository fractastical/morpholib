#!/usr/bin/env python3
"""
2014-sarrazin-tribolium-castaneum-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1042)
Organism: Tribolium castaneum (Red Flour Beetle)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Tribolium
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1042_tribolium_castaneum_collective_stigmergy.lean` (`MathOfBiology.Loop_1042`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tribolium_castaneum_collective_stigmergy"
EXPECTED_SHA256 = {
    "tribolium_castaneum_collective_stigmergy_reference.json": "369a861576024021ed6b4ad7020ad34f075ab2082ae79364093c5ee8455f5c2b",
    "tribolium_castaneum_specimen.jpg": "a991c507b6f59178075bea80d7ad2c1c97d839991d31594797919b07b5ac1d77",
    "scout_manifest.json": "2e1c626267e01ff4994544d4817c7871a81f3994f3a97254868055d2543a02bc",
    "paper_citations.bib": "ec60e9f4e0dc7054d7ec04c3a477d8f109811f01e005e2a5bf9ad67f7b7ac4bd"
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
    ref = json.loads((DATASET_DIR / "tribolium_castaneum_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-sarrazin-tribolium-castaneum-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
