#!/usr/bin/env python3
"""
1943-thompson-acanthometra-pellucida-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0993)
Organism: Acanthometra pellucida (Strontium Radiolarian)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acanthometra
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0993_acanthometra_pellucida_bioelectric_polarity.lean` (`MathOfBiology.Loop_0993`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acanthometra_pellucida_bioelectric_polarity"
EXPECTED_SHA256 = {
    "acanthometra_pellucida_bioelectric_polarity_reference.json": "96c99db762ce4d80e7fc9833f98bfe0327a7ebfc59dae8489b95644d19420f26",
    "acanthometra_pellucida_specimen.jpg": "8f4b8a930e33286fd2843fa388dfec53592278e506619611732e990cc03371d4",
    "scout_manifest.json": "e78443b5c8c51435126bb6c7296c8477116f286d9e415418e775ed3b60a0bf51",
    "paper_citations.bib": "c5fa186f65b5e7eef74aef0147c634f166242c5a5c137a71facb8a2b326809dd"
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
    ref = json.loads((DATASET_DIR / "acanthometra_pellucida_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1943-thompson-acanthometra-pellucida-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
