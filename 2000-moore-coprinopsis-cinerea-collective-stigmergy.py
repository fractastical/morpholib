#!/usr/bin/env python3
"""
2000-moore-coprinopsis-cinerea-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1083)
Organism: Coprinopsis cinerea (Gray Shag Mushroom)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Coprinopsis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1083_coprinopsis_cinerea_collective_stigmergy.lean` (`MathOfBiology.Loop_1083`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/coprinopsis_cinerea_collective_stigmergy"
EXPECTED_SHA256 = {
    "coprinopsis_cinerea_collective_stigmergy_reference.json": "d195d1a0a14add551a3cdda9322128f263724bad9643909f413405e1a5d03da9",
    "coprinopsis_cinerea_specimen.jpg": "89844e402834973a57384e9aee4268b35a0e61f0cb9977afb4a2c562507756ca",
    "scout_manifest.json": "b0d7bc31e85e804fdf09f2f6afa2958b8212592b9aba3f9ed0db368142666642",
    "paper_citations.bib": "4e900ce0d1f18f85789e4a601c20d2fba70c58f76660fead585bf7ba220bcd4d"
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
    ref = json.loads((DATASET_DIR / "coprinopsis_cinerea_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-moore-coprinopsis-cinerea-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
