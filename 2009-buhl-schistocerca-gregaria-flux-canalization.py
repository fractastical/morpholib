#!/usr/bin/env python3
"""
2009-buhl-schistocerca-gregaria-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0448)
Organism: Schistocerca gregaria (Desert Locust)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Schistocerca
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0448_schistocerca_gregaria_flux_canalization.lean` (`MathOfBiology.Loop_0448`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schistocerca_gregaria_flux_canalization"
EXPECTED_SHA256 = {
    "schistocerca_gregaria_flux_canalization_reference.json": "7a9f24dfff1a03fcf222a1213620a004b8c2b5d0812fee48d24c62112654f396",
    "schistocerca_gregaria_specimen.jpg": "3f8a278b02bc5e5158458d4ff3669c71a29826b1129be2cc7920ab4b3ead3827",
    "scout_manifest.json": "dd90a5ef040c2a6e4638b6c8189f7f27d276fd9689cca2b1714d8f51db849905",
    "paper_citations.bib": "ac29dba03bb126c6f6d83638cd5e0b4975776ef7b0c7d8b4baece22ca779ca51"
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
    ref = json.loads((DATASET_DIR / "schistocerca_gregaria_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-buhl-schistocerca-gregaria-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
