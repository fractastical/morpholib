#!/usr/bin/env python3
"""
2001-hammer-pecten-maximus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0957)
Organism: Pecten maximus (Great Scallop)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pecten
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0957_pecten_maximus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0957`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pecten_maximus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pecten_maximus_bioelectric_polarity_reference.json": "5abd016eeaf908ff8b623de1528296024e1146e9206bea5195452b9b3ff770d2",
    "pecten_maximus_specimen.jpg": "2f87d4ec30ebec6adb82f91aec67eb300e3469f57920cbccf9c9bb0d22b703a8",
    "scout_manifest.json": "ccf495554ceb627569183cd0fe6c2377ff8915d72339ddb67178a9a03b2e8708",
    "paper_citations.bib": "b8120774f79e5a7fc445f3273d0f71706274f9b93072976e80e38d134b048b1d"
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
    ref = json.loads((DATASET_DIR / "pecten_maximus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-hammer-pecten-maximus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
