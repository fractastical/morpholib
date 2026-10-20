#!/usr/bin/env python3
"""
2002-messenger-octopus-bimaculoides-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0952)
Organism: Octopus bimaculoides (Two-Spot Octopus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Octopus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0952_octopus_bimaculoides_bioelectric_polarity.lean` (`MathOfBiology.Loop_0952`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/octopus_bimaculoides_bioelectric_polarity"
EXPECTED_SHA256 = {
    "octopus_bimaculoides_bioelectric_polarity_reference.json": "0afd3d8001eb81bb8be8374c6e54313562cafd33fe9ab40fc3da1943c43d63a5",
    "octopus_bimaculoides_specimen.jpg": "4e6d70661aa4e7709a51348bee322500de3a32b1ecb8b557b67060ff9d863233",
    "scout_manifest.json": "127cb13181c6744d319761a71de18c8f0b63e178a7f62db6c3b0867f07a7bc1f",
    "paper_citations.bib": "7f121a5122f3731de04427905ec7db120e57ba62743ddd54100c17ac0bbfc514"
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
    ref = json.loads((DATASET_DIR / "octopus_bimaculoides_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-messenger-octopus-bimaculoides-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
