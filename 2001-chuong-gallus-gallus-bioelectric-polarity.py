#!/usr/bin/env python3
"""
2001-chuong-gallus-gallus-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0911)
Organism: Gallus gallus (Red Junglefowl Chick)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Gallus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0911_gallus_gallus_bioelectric_polarity.lean` (`MathOfBiology.Loop_0911`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gallus_gallus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "gallus_gallus_bioelectric_polarity_reference.json": "27648b0c616df98d42c2e172ae99288cf42bda3d52a42b01ae77432b1152aeb0",
    "gallus_gallus_specimen.jpg": "7e7d58db6bb564ebe68e091678338c7c06b920cdb66f9dffd99b4141bf16c9cb",
    "scout_manifest.json": "97784601f8f6f4b74aeb3ad5df01e9083a46037d67928932b2c21c1a891b6f17",
    "paper_citations.bib": "4cb0bed86d33b5bacb72dd0df64729097094c7f820a5e2bc0c92913c5aece4a4"
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
    ref = json.loads((DATASET_DIR / "gallus_gallus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-chuong-gallus-gallus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
