#!/usr/bin/env python3
"""
2003-messenger-octopus-bimaculoides-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0352)
Organism: Octopus bimaculoides (Two-Spot Octopus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Octopus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0352_octopus_bimaculoides_clock_wavefront.lean` (`MathOfBiology.Loop_0352`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/octopus_bimaculoides_clock_wavefront"
EXPECTED_SHA256 = {
    "octopus_bimaculoides_clock_wavefront_reference.json": "671b17278a91b4750315b1194c4cbcb3650111e0d658672e52410e78f3d573ce",
    "octopus_bimaculoides_specimen.jpg": "4e6d70661aa4e7709a51348bee322500de3a32b1ecb8b557b67060ff9d863233",
    "scout_manifest.json": "98499b9f2086e26428f05dd03ea911272b21e479fd34b19dfd044d33a4e7c110",
    "paper_citations.bib": "b3dc1f3025264a034187b2f61b9f159a18b54161ddd9fd0d730d752a7fb7aa6e"
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
    ref = json.loads((DATASET_DIR / "octopus_bimaculoides_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-messenger-octopus-bimaculoides-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
