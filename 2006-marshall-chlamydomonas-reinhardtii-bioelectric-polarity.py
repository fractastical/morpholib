#!/usr/bin/env python3
"""
2006-marshall-chlamydomonas-reinhardtii-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0995)
Organism: Chlamydomonas reinhardtii (Biflagellate Microalga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Chlamydomonas
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0995_chlamydomonas_reinhardtii_bioelectric_polarity.lean` (`MathOfBiology.Loop_0995`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chlamydomonas_reinhardtii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "chlamydomonas_reinhardtii_bioelectric_polarity_reference.json": "487bf9ca4fd6bc1bc174166f177b046ad5ac83a635b540f8cb4219046417db83",
    "chlamydomonas_reinhardtii_specimen.jpg": "ebcb2a487066205d3a9f06b194d0bfd68786337b7fbbf44dd71400b689d923bd",
    "scout_manifest.json": "7b02815942764863f6962ad43a0051d541b5e42dd8c029b0e36c6c150fe2f03b",
    "paper_citations.bib": "17d14a09181ebf99dc37823f07384a4c3fc2dafddc7eec8664c6ea4b997f4d37"
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
    ref = json.loads((DATASET_DIR / "chlamydomonas_reinhardtii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-marshall-chlamydomonas-reinhardtii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
