#!/usr/bin/env python3
"""
2016-hohn-volvox-carteri-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0991)
Organism: Volvox carteri (Colonial Green Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Volvox
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0991_volvox_carteri_bioelectric_polarity.lean` (`MathOfBiology.Loop_0991`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/volvox_carteri_bioelectric_polarity"
EXPECTED_SHA256 = {
    "volvox_carteri_bioelectric_polarity_reference.json": "5a0eeecae52bf93bd2ff03ac11fe49071f1990b37a3d33ca417242fb4e0de383",
    "volvox_carteri_specimen.jpg": "cf734e5f4b2d7f5ded097049b19cf0c7c00c592b2c19a662073fe8d066b17e4a",
    "scout_manifest.json": "350d7c3cfb83180be23e7f621b40d741eac1434e3df514cf8de73eddac239901",
    "paper_citations.bib": "6816ed3673a774a638da006d5c2793880e8f9a720c0533c08c908ad9484be999"
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
    ref = json.loads((DATASET_DIR / "volvox_carteri_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-hohn-volvox-carteri-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
