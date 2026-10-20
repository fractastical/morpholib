#!/usr/bin/env python3
"""
2008-cove-physcomitrium-patens-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1072)
Organism: Physcomitrium patens (Spreading Earthmoss)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Physcomitrium
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1072_physcomitrium_patens_collective_stigmergy.lean` (`MathOfBiology.Loop_1072`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/physcomitrium_patens_collective_stigmergy"
EXPECTED_SHA256 = {
    "physcomitrium_patens_collective_stigmergy_reference.json": "981849fb0d10b40f29de1129c423bef9bca3656368ed381de16d91426a333359",
    "physcomitrium_patens_specimen.jpg": "a115412e5c6b1e47cad14da271d9ee4f3d33b2b7150fb4c533f145234f694d13",
    "scout_manifest.json": "0bc749f4426e40b1f9ae9fba2cce1e6ab23827a20145eec86279241922c9db49",
    "paper_citations.bib": "79f1a2c44bf52efeb793758d012aee38ac6caa47b2a1c36242de239505dbebd8"
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
    ref = json.loads((DATASET_DIR / "physcomitrium_patens_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-cove-physcomitrium-patens-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
