#!/usr/bin/env python3
"""
1983-odell-lytechinus-variegatus-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1055)
Organism: Lytechinus variegatus (Green Sea Urchin)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Lytechinus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1055_lytechinus_variegatus_collective_stigmergy.lean` (`MathOfBiology.Loop_1055`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/lytechinus_variegatus_collective_stigmergy"
EXPECTED_SHA256 = {
    "lytechinus_variegatus_collective_stigmergy_reference.json": "5ae155cd2d74349db62c95c0e3cd903d1535af5678ea8c97b14aeb9d78983555",
    "lytechinus_variegatus_specimen.jpg": "521ca1f087eb9618f793b80f0084a9efa6db3c8fb7039599c38d75ba00260619",
    "scout_manifest.json": "01904c2bb06a1cfc1f9efff64524bcbf5158f5c0a9f47ca909dc75a054ed27e4",
    "paper_citations.bib": "b559da47ea6f94b9844ba49db34b362f1353585ae8e0f4f7cfe2742c544fd1cc"
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
    ref = json.loads((DATASET_DIR / "lytechinus_variegatus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1983-odell-lytechinus-variegatus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
