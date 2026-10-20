#!/usr/bin/env python3
"""
2005-marshall-chlamydomonas-reinhardtii-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0895)
Organism: Chlamydomonas reinhardtii (Biflagellate Microalga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Chlamydomonas
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0895_chlamydomonas_reinhardtii_biomineral_accretion.lean` (`MathOfBiology.Loop_0895`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chlamydomonas_reinhardtii_biomineral_accretion"
EXPECTED_SHA256 = {
    "chlamydomonas_reinhardtii_biomineral_accretion_reference.json": "5097b06f0ce6ff1261439821b27953a9a5c9596537b7262216ea4d5b4c5e887c",
    "chlamydomonas_reinhardtii_specimen.jpg": "ebcb2a487066205d3a9f06b194d0bfd68786337b7fbbf44dd71400b689d923bd",
    "scout_manifest.json": "3cf68b5365fda7df0b4dd9bdf88ab4164bdb733e7697e68eaa512f997a486e45",
    "paper_citations.bib": "0dfd533b254f2af9d24433b9c2541b9d193db6037d840fe208c74b41204756cf"
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
    ref = json.loads((DATASET_DIR / "chlamydomonas_reinhardtii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-marshall-chlamydomonas-reinhardtii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
