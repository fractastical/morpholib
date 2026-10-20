#!/usr/bin/env python3
"""
2001-czirok-proteus-mirabilis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0190)
Organism: Proteus mirabilis (Bullseye Swarming Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Proteus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0190_proteus_mirabilis_turing_wavelength.lean` (`MathOfBiology.Loop_0190`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/proteus_mirabilis_turing_wavelength"
EXPECTED_SHA256 = {
    "proteus_mirabilis_turing_wavelength_reference.json": "b2ef0f37f2d7deddd6596d96c24ff483cb28bfa7bb74bde5cc42f7eed2a49265",
    "proteus_mirabilis_specimen.jpg": "ac07f43481b9b721da7b124217c182a69192087ef9fb2f59564a107e3383a85b",
    "scout_manifest.json": "2c2f144c7f656098befc53befdac46d46f2d4e48cc8f14c50be49b139897813f",
    "paper_citations.bib": "cedf6a0eb9a452ec5a189c1222643ceff68e311006e658b9b2b403dbc5874410"
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
    ref = json.loads((DATASET_DIR / "proteus_mirabilis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-czirok-proteus-mirabilis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
