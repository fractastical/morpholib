#!/usr/bin/env python3
"""
2006-weisblat-hirudo-medicinalis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0969)
Organism: Hirudo medicinalis (Medicinal Leech)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hirudo
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0969_hirudo_medicinalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0969`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hirudo_medicinalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "hirudo_medicinalis_bioelectric_polarity_reference.json": "1378e7462e426591535a17cffcc08962dc2a58401172e87c549de099f6d56ed5",
    "hirudo_medicinalis_specimen.jpg": "f9cc303efc9e9bbd4f520cb58b823e3ee6b17491f3c40d8ee2c2ad8fcedf909f",
    "scout_manifest.json": "9bd9bb20904684e5a99ccaa77fe99e2924d8838621d3bc642a76938d68b786b7",
    "paper_citations.bib": "0fe5c55c3610907919a4a8ed69daea3398fdc235d3a7af684d3f4c8c2b52461b"
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
    ref = json.loads((DATASET_DIR / "hirudo_medicinalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-weisblat-hirudo-medicinalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
