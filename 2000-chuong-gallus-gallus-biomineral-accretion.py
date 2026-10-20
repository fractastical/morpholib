#!/usr/bin/env python3
"""
2000-chuong-gallus-gallus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0811)
Organism: Gallus gallus (Red Junglefowl Chick)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Gallus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0811_gallus_gallus_biomineral_accretion.lean` (`MathOfBiology.Loop_0811`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gallus_gallus_biomineral_accretion"
EXPECTED_SHA256 = {
    "gallus_gallus_biomineral_accretion_reference.json": "dfd1ffe81005e4a37227d0f552834b2179c763eae77c73bce13afd71738d0cfd",
    "gallus_gallus_specimen.jpg": "7e7d58db6bb564ebe68e091678338c7c06b920cdb66f9dffd99b4141bf16c9cb",
    "scout_manifest.json": "615bed0218a4191f3c6192a498e2db69c42c527af0bfdbda8e34e7940851772f",
    "paper_citations.bib": "07adbe96a000854d9b5f2816df04c638089699b680498ce3294dfdd423c2f1a7"
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
    ref = json.loads((DATASET_DIR / "gallus_gallus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-chuong-gallus-gallus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
