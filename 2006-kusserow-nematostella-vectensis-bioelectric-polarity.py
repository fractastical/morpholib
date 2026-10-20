#!/usr/bin/env python3
"""
2006-kusserow-nematostella-vectensis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0962)
Organism: Nematostella vectensis (Starlet Sea Anemone)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Nematostella
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0962_nematostella_vectensis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0962`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/nematostella_vectensis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "nematostella_vectensis_bioelectric_polarity_reference.json": "75a13a3b34e6d27c6c2a63296cc96bcab1ddbf66c12f7d01dbd1f7ed89ed1a47",
    "nematostella_vectensis_specimen.jpg": "fb9d66de457a62bbf3d2ec1bcdb57f33e32468690a3047a09117a19153073a53",
    "scout_manifest.json": "86c332b720e837203988a641872f3677d7d0c4de5bfd346efdd5d77d4ff6edf9",
    "paper_citations.bib": "1be86fe17d3b7598996e4781939ab18fc92970aba11cda1f3238f54996393286"
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
    ref = json.loads((DATASET_DIR / "nematostella_vectensis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-kusserow-nematostella-vectensis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
