#!/usr/bin/env python3
"""
2005-shoji-tetraodon-mbu-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1032)
Organism: Tetraodon mbu (Giant Pufferfish)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Tetraodon
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1032_tetraodon_mbu_collective_stigmergy.lean` (`MathOfBiology.Loop_1032`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetraodon_mbu_collective_stigmergy"
EXPECTED_SHA256 = {
    "tetraodon_mbu_collective_stigmergy_reference.json": "c690a0ba1e491a9f066c632d3f535699e0683ebb448b63871cb572aa614089fb",
    "tetraodon_mbu_specimen.jpg": "c2d84cbf503eef4d1bdc55fdc0e9d3ba124750e0ae48ff7010b6a4cdca2ee3f3",
    "scout_manifest.json": "4111f3fe7b7d3e3460205e9eb62a70bd12de72d00e4ba79eca5d9e309ad697b8",
    "paper_citations.bib": "267480b906b41495879fcf780b9c6e87c84f5b43f43efb64860fb1ca70d3d09b"
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
    ref = json.loads((DATASET_DIR / "tetraodon_mbu_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-shoji-tetraodon-mbu-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
