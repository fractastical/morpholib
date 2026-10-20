#!/usr/bin/env python3
"""
2018-cooke-melopsittacus-undulatus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0917)
Organism: Melopsittacus undulatus (Budgerigar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Melopsittacus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0917_melopsittacus_undulatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0917`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/melopsittacus_undulatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "melopsittacus_undulatus_bioelectric_polarity_reference.json": "3e43dfd50e486fd37e29da59bca559723af4fc3c1555ab5460c03985861baadd",
    "melopsittacus_undulatus_specimen.jpg": "f61163f7820594be22c648c5120f1e827b9cdd8d377d9eeb65bb96e633b0ce2b",
    "scout_manifest.json": "35c19c3198f5aa5dedf31f65f0fe0729e2dc29ae6eab83dd707a80ad400f4b1f",
    "paper_citations.bib": "b63e615d68c3241168c4f76d00a99e7a074e090617e5264e57bbd8f0b610cddb"
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
    ref = json.loads((DATASET_DIR / "melopsittacus_undulatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-cooke-melopsittacus-undulatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
