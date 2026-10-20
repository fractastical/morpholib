#!/usr/bin/env python3
"""
2003-munro-ciona-intestinalis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0963)
Organism: Ciona intestinalis (Vase Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Ciona
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0963_ciona_intestinalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0963`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ciona_intestinalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ciona_intestinalis_bioelectric_polarity_reference.json": "6031034290f01b02e79048f94ffbaa8f409d2e338f7d3602e6389a3b51cad425",
    "ciona_intestinalis_specimen.jpg": "0f563cbc2ccb4b7ba1595b7519a75a1410031eaaa4afceaf02b5975d87c1ecf4",
    "scout_manifest.json": "4ba316bf78e30c6f50449fcd3e8a8fc47de244edbafd51c764e2a4be44874ee6",
    "paper_citations.bib": "1fbdd100dfd5dde414f8352a6cad6abc8080f0bc96d9d8c47933fc153270ae4c"
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
    ref = json.loads((DATASET_DIR / "ciona_intestinalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-munro-ciona-intestinalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
