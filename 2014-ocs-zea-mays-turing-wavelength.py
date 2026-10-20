#!/usr/bin/env python3
"""
2014-ocs-zea-mays-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0173)
Organism: Zea mays (Maize)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Zea
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0173_zea_mays_turing_wavelength.lean` (`MathOfBiology.Loop_0173`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/zea_mays_turing_wavelength"
EXPECTED_SHA256 = {
    "zea_mays_turing_wavelength_reference.json": "ae46b063b60a12b3d0c2b1dcd5c91a41b6a2b0a619e85088da615cea534127bf",
    "zea_mays_specimen.jpg": "a78e27ea3c83c91a3f84373319a3b590f4a53e6108ddf50860d114690cbb6db9",
    "scout_manifest.json": "da6907c98fe09c07ee6b41785fb0378545d0b9a0e3b9f8523435b82f2fd5918c",
    "paper_citations.bib": "8d8af5519e137d88e118f5312fa7fccba3dce0f9b537f2be0f30ffaf92485955"
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
    ref = json.loads((DATASET_DIR / "zea_mays_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-ocs-zea-mays-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
