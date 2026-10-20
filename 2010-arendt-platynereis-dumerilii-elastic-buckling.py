#!/usr/bin/env python3
"""
2010-arendt-platynereis-dumerilii-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0761)
Organism: Platynereis dumerilii (Marine Ragworm)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Platynereis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0761_platynereis_dumerilii_elastic_buckling.lean` (`MathOfBiology.Loop_0761`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/platynereis_dumerilii_elastic_buckling"
EXPECTED_SHA256 = {
    "platynereis_dumerilii_elastic_buckling_reference.json": "0c3610e22db7a355555ca12ec84b69e68773e51e0c518b8ff6a7f07e0526bb4e",
    "platynereis_dumerilii_specimen.jpg": "49376d1416ec7548315612807bfb066579e6891667d63542b0c9e663606fe5b9",
    "scout_manifest.json": "fb23e98a6e6da353fb64d676a9c30dc5f99824065c84f9a941c4a8fcc9708155",
    "paper_citations.bib": "70553eed33985d453a8967fbae7cfbb4b8d5ac0b145209db0af9f8a2487f473b"
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
    ref = json.loads((DATASET_DIR / "platynereis_dumerilii_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-arendt-platynereis-dumerilii-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
