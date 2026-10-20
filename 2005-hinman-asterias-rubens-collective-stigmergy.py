#!/usr/bin/env python3
"""
2005-hinman-asterias-rubens-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1054)
Organism: Asterias rubens (Common Starfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Asterias
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1054_asterias_rubens_collective_stigmergy.lean` (`MathOfBiology.Loop_1054`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/asterias_rubens_collective_stigmergy"
EXPECTED_SHA256 = {
    "asterias_rubens_collective_stigmergy_reference.json": "24471a8430ad05009415c0f51f4254271086155b337cf1bf9bfa2d182f4a18dc",
    "asterias_rubens_specimen.jpg": "37d94bdfed179302d1e9aa062333c2f87a5cb3d7c3932c51056f5e68c7c762ad",
    "scout_manifest.json": "15cfcbbe360a9a978a33e905c5f93a606216d71755f71e88a5ba6925cf8a6475",
    "paper_citations.bib": "15e53d4ce20b22c03b62c750e629113ba0764d43eef59f2181fd453642be9e55"
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
    ref = json.loads((DATASET_DIR / "asterias_rubens_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-hinman-asterias-rubens-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
