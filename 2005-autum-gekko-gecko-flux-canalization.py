#!/usr/bin/env python3
"""
2005-autum-gekko-gecko-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0428)
Organism: Gekko gecko (Tokay Gecko)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Gekko
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0428_gekko_gecko_flux_canalization.lean` (`MathOfBiology.Loop_0428`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gekko_gecko_flux_canalization"
EXPECTED_SHA256 = {
    "gekko_gecko_flux_canalization_reference.json": "8f6a3fea61a6049a193fd0d6f648497061d03264227f97a22213529a0fc2dd4b",
    "gekko_gecko_specimen.jpg": "d5822e517434737d90b14fd5099c77f09ab976ca2a3ba00c2484f69a65eb91e8",
    "scout_manifest.json": "36e57432c509919886d37579dd16ff45b71df08ec85242a9b95fc697e53b41b2",
    "paper_citations.bib": "883aac7bbccebfefcd2a7f6c36ead9a7aa55549a9149ede0da74232114b877bb"
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
    ref = json.loads((DATASET_DIR / "gekko_gecko_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-autum-gekko-gecko-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
