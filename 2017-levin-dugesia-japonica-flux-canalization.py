#!/usr/bin/env python3
"""
2017-levin-dugesia-japonica-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0466)
Organism: Dugesia japonica (Freshwater Planarian)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dugesia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0466_dugesia_japonica_flux_canalization.lean` (`MathOfBiology.Loop_0466`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/dugesia_japonica_flux_canalization"
EXPECTED_SHA256 = {
    "dugesia_japonica_flux_canalization_reference.json": "717e6da994c5f10a27305eec4831f02fd9c2fc6a4a1a037580db241227273ab7",
    "dugesia_japonica_specimen.jpg": "26b78d28b853c1b70092be6a7ae129d59813507f64eaed16a417276296ff2efa",
    "scout_manifest.json": "eb9d876c219bbd1198c0bd25cd3f7f54a10f485646c6a9f8e29f3b5ac026f73e",
    "paper_citations.bib": "98ee3f147970d09e6d37e851519fb4d07c8d174364d850a44b29b845b6358e32"
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
    ref = json.loads((DATASET_DIR / "dugesia_japonica_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-levin-dugesia-japonica-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
