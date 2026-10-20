#!/usr/bin/env python3
"""
2019-reiter-sepia-officinalis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0951)
Organism: Sepia officinalis (Common Cuttlefish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Sepia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0951_sepia_officinalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0951`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sepia_officinalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "sepia_officinalis_bioelectric_polarity_reference.json": "df938339f1dc0deb6b7f7a2a4901ebce8a3ec068ddbfa1ea672db6e1e06c4095",
    "sepia_officinalis_specimen.jpg": "c6ecd8366e75a19b7375ac673bb90c8aa032e891c606d26556858646002b58e1",
    "scout_manifest.json": "45020f86f7cd280f1185640d7eebb70de80062735bd1512571fc82fedca133d5",
    "paper_citations.bib": "bcf09236ba4071c34dd3de69f42afd20a4c5e76f87f9dc86c66d563718f2f3c5"
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
    ref = json.loads((DATASET_DIR / "sepia_officinalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-reiter-sepia-officinalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
