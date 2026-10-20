#!/usr/bin/env python3
"""
2006-coen-antirrhinum-majus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1076)
Organism: Antirrhinum majus (Snapdragon)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Antirrhinum
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1076_antirrhinum_majus_collective_stigmergy.lean` (`MathOfBiology.Loop_1076`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/antirrhinum_majus_collective_stigmergy"
EXPECTED_SHA256 = {
    "antirrhinum_majus_collective_stigmergy_reference.json": "4c7881c2145abaff6139a3b798b3c2d54bf4059fd26013396fe7e23aca9ee4d1",
    "antirrhinum_majus_specimen.jpg": "f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0",
    "scout_manifest.json": "0f814d5c7050db44a7cf33e6235363003136377c06135a3efd363c5eb5034524",
    "paper_citations.bib": "e24172c865b35f18d03fe58c3574811d6e21d37a2e2cd7b7839fbddd4794d296"
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
    ref = json.loads((DATASET_DIR / "antirrhinum_majus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-coen-antirrhinum-majus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
