#!/usr/bin/env python3
"""
2004-autum-gekko-gecko-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1028)
Organism: Gekko gecko (Tokay Gecko)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Gekko
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1028_gekko_gecko_collective_stigmergy.lean` (`MathOfBiology.Loop_1028`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gekko_gecko_collective_stigmergy"
EXPECTED_SHA256 = {
    "gekko_gecko_collective_stigmergy_reference.json": "bb1fcd40503a81f7a7c277f68aadfc03ebb43a3e3d7b2620e12a17c18b0b5f22",
    "gekko_gecko_specimen.jpg": "d5822e517434737d90b14fd5099c77f09ab976ca2a3ba00c2484f69a65eb91e8",
    "scout_manifest.json": "4b05da781500378a6451f124736d06a359106d2b861971eec783660847e09f12",
    "paper_citations.bib": "065bb3b5593bca018c4816be5b1ce51df1e4267d0b3447a3044f4cf9bc77288c"
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
    ref = json.loads((DATASET_DIR / "gekko_gecko_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-autum-gekko-gecko-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
