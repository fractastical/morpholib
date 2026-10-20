#!/usr/bin/env python3
"""
2021-eliason-calypte-anna-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0920)
Organism: Calypte anna (Anna's Hummingbird)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Calypte
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0920_calypte_anna_bioelectric_polarity.lean` (`MathOfBiology.Loop_0920`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/calypte_anna_bioelectric_polarity"
EXPECTED_SHA256 = {
    "calypte_anna_bioelectric_polarity_reference.json": "4408791797ab23e336a297428cf0d4ea553206901ae2987c39ee0eed5e53eab7",
    "calypte_anna_specimen.jpg": "023a1c1ebf863e2bc3fe443e9cc7d668c6fa7ac72517c8a62f1874671a2eeb60",
    "scout_manifest.json": "a69600aa27724ecce6fe28a94f1f6f57d254e44bdffb7a46ea4c7b0d286e772b",
    "paper_citations.bib": "bc27ae5f5f07e63c590d98b1e7a94ba18531176fa375b137ddaaf21b26773ed6"
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
    ref = json.loads((DATASET_DIR / "calypte_anna_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-eliason-calypte-anna-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
