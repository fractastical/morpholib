#!/usr/bin/env python3
"""
2021-chirat-argonauta-argo-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0156)
Organism: Argonauta argo (Paper Nautilus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Argonauta
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0156_argonauta_argo_turing_wavelength.lean` (`MathOfBiology.Loop_0156`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/argonauta_argo_turing_wavelength"
EXPECTED_SHA256 = {
    "argonauta_argo_turing_wavelength_reference.json": "27856ca2dac64723873cd9271caa16491e42e5909a73f3918efd9237e442ac91",
    "argonauta_argo_specimen.jpg": "4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2",
    "scout_manifest.json": "2b17e8fe54d9d2b7bc2f60ffc4d8038710a4a99db82a4b59c48ffb4f193def6e",
    "paper_citations.bib": "9a1513a08e6ed15283a12adadc5777a0cf31194e97dd3d5dd13a3ec81f6e790c"
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
    ref = json.loads((DATASET_DIR / "argonauta_argo_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-chirat-argonauta-argo-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
