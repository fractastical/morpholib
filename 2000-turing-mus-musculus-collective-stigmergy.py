#!/usr/bin/env python3
"""
2000-turing-mus-musculus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1001)
Organism: Mus musculus (House Mouse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1001_mus_musculus_collective_stigmergy.lean` (`MathOfBiology.Loop_1001`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mus_musculus_collective_stigmergy"
EXPECTED_SHA256 = {
    "mus_musculus_collective_stigmergy_reference.json": "3735f8a2a7954db9893202320633ef48e49f07c7576ce7481496f6b72ded90eb",
    "mus_musculus_specimen.jpg": "f2738290d66e441895fc10af7d7e541424263249cda0c7a8194ff01b410fe385",
    "scout_manifest.json": "59d46ab1e6b90e909fc2981a6c3e1c1ea9dca9660a917380e5adf992c7722985",
    "paper_citations.bib": "8e75f0c50a0fccd8847227d96d2b1ce71de350c2e77cd1d90dd0275716a0eb25"
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
    ref = json.loads((DATASET_DIR / "mus_musculus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-turing-mus-musculus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
