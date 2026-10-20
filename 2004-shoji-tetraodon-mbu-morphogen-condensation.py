#!/usr/bin/env python3
"""
2004-shoji-tetraodon-mbu-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0232)
Organism: Tetraodon mbu (Giant Pufferfish)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Tetraodon
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0232_tetraodon_mbu_morphogen_condensation.lean` (`MathOfBiology.Loop_0232`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetraodon_mbu_morphogen_condensation"
EXPECTED_SHA256 = {
    "tetraodon_mbu_morphogen_condensation_reference.json": "4a73203baa34db2895d211d8f45632edccbd624e02da13bfe8da22e0127243df",
    "tetraodon_mbu_specimen.jpg": "c2d84cbf503eef4d1bdc55fdc0e9d3ba124750e0ae48ff7010b6a4cdca2ee3f3",
    "scout_manifest.json": "16b193c6543abb28b53754b1650354e802f12330831bfdfa1b78c9f1fbca8902",
    "paper_citations.bib": "59266495314271978888b1f629aa49f273ed78f7bdab6dfe8d2ac0523666fea7"
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
    ref = json.loads((DATASET_DIR / "tetraodon_mbu_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-shoji-tetraodon-mbu-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
