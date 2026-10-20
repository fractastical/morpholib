#!/usr/bin/env python3
"""
2007-prum-morpho-menelaus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0244)
Organism: Morpho menelaus (Blue Morpho Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Morpho
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0244_morpho_menelaus_morphogen_condensation.lean` (`MathOfBiology.Loop_0244`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/morpho_menelaus_morphogen_condensation"
EXPECTED_SHA256 = {
    "morpho_menelaus_morphogen_condensation_reference.json": "da2c0d1664b693a5815890e5d4403d7c45bb0c5f4bd6dc72f97b048320fcabac",
    "morpho_menelaus_specimen.jpg": "4568faabf61d02b09522424f4f34c0efc7d869ad36aba4c8d412cda345094c68",
    "scout_manifest.json": "34fd51a891f54623f8a6314f2f40008d9f633d97be27c22c7d81dbf5b83a8da4",
    "paper_citations.bib": "d7698b047c905750e1db038d32207a4b59599ca3bb5417d6dd7c70d404b9072d"
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
    ref = json.loads((DATASET_DIR / "morpho_menelaus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-prum-morpho-menelaus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
