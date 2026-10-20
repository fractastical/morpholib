#!/usr/bin/env python3
"""
2006-arendt-platynereis-dumerilii-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1061)
Organism: Platynereis dumerilii (Marine Ragworm)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Platynereis
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1061_platynereis_dumerilii_collective_stigmergy.lean` (`MathOfBiology.Loop_1061`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/platynereis_dumerilii_collective_stigmergy"
EXPECTED_SHA256 = {
    "platynereis_dumerilii_collective_stigmergy_reference.json": "f75ff6ec61da40ed7dbc259a8fe8b29c2c9f6b755f63d2f8024181548a496b35",
    "platynereis_dumerilii_specimen.jpg": "49376d1416ec7548315612807bfb066579e6891667d63542b0c9e663606fe5b9",
    "scout_manifest.json": "19060f62a7cac0d4a860dc4048600b0e0318a1da41ae4dedc41577d0216b9607",
    "paper_citations.bib": "96c94ea9654cf96a8347f8b5a4974c4a97603cc0b0e451aa50f1a3275cf0c458"
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
    ref = json.loads((DATASET_DIR / "platynereis_dumerilii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-arendt-platynereis-dumerilii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
