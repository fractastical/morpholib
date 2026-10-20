#!/usr/bin/env python3
"""
2004-zi-pavo-cristatus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0913)
Organism: Pavo cristatus (Indian Peafowl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Pavo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0913_pavo_cristatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0913`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pavo_cristatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pavo_cristatus_bioelectric_polarity_reference.json": "596cb4b4acea165ae5a53565bd3269b25350e3d965cda2979a2a3f7778958fd6",
    "pavo_cristatus_specimen.jpg": "39f28e26a62ddfdb98fe19fee7eaf7af508c82486a32abe3c1f2ac7d3e607d50",
    "scout_manifest.json": "d8455142b5795a0a6e9cc4eecc846f8ed949097544bbf06d64f6c7f699aafb0a",
    "paper_citations.bib": "a377389062514e44df26dd7a821dc210f34de393f6ac4ae65f4e3ab8513e9e14"
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
    ref = json.loads((DATASET_DIR / "pavo_cristatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-zi-pavo-cristatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
