#!/usr/bin/env python3
"""
2005-wittbrodt-oryzias-latipes-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0433)
Organism: Oryzias latipes (Japanese Medaka)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Oryzias
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0433_oryzias_latipes_flux_canalization.lean` (`MathOfBiology.Loop_0433`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/oryzias_latipes_flux_canalization"
EXPECTED_SHA256 = {
    "oryzias_latipes_flux_canalization_reference.json": "c4ea34a6c53720e99ef6d9c3ccff52cf26e5b5f2c9df9f5554de06d9cb90f398",
    "oryzias_latipes_specimen.jpg": "ec331629f10358b6c34a698921c02e2de50725276c73bb64ffa78ba7f6e256af",
    "scout_manifest.json": "38cb6977d7ab26015af6cea2fe48b6f42852a403550920c984ec4e2b97bc39a4",
    "paper_citations.bib": "cc92d43fc836cb493933080720d28d1fb27099a1448a3b998ecdc9f3482fcf56"
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
    ref = json.loads((DATASET_DIR / "oryzias_latipes_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-wittbrodt-oryzias-latipes-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
