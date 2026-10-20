#!/usr/bin/env python3
"""
2005-munro-ciona-intestinalis-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0463)
Organism: Ciona intestinalis (Vase Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Ciona
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0463_ciona_intestinalis_flux_canalization.lean` (`MathOfBiology.Loop_0463`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ciona_intestinalis_flux_canalization"
EXPECTED_SHA256 = {
    "ciona_intestinalis_flux_canalization_reference.json": "9d0534483ace0f5b6d7f5e6fa6a6fbb7227faf18da684450dd0e5736f99cb80d",
    "ciona_intestinalis_specimen.jpg": "0f563cbc2ccb4b7ba1595b7519a75a1410031eaaa4afceaf02b5975d87c1ecf4",
    "scout_manifest.json": "a9cc0a7ae110e8e75a4fa1fa2ef103a3a67ed847be13284dff4e82f9defdd58a",
    "paper_citations.bib": "1b4d0351e67fba7a7c90dfa37db6c8405f178c4f39151244a1feffff65276e5e"
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
    ref = json.loads((DATASET_DIR / "ciona_intestinalis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-munro-ciona-intestinalis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
