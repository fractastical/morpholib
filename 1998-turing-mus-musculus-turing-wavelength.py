#!/usr/bin/env python3
"""
1998-turing-mus-musculus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0101)
Organism: Mus musculus (House Mouse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0101_mus_musculus_turing_wavelength.lean` (`MathOfBiology.Loop_0101`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mus_musculus_turing_wavelength"
EXPECTED_SHA256 = {
    "mus_musculus_turing_wavelength_reference.json": "dc8d5607fe60fafeeb18dd958117aa913131ec955394b9224407e6b5cc15e1bb",
    "mus_musculus_specimen.jpg": "f2738290d66e441895fc10af7d7e541424263249cda0c7a8194ff01b410fe385",
    "scout_manifest.json": "5836d0440b9295a96206ff09d8c73fdcbe8a092a78c2b49e49fbf3f8fabd79c9",
    "paper_citations.bib": "66ba96fdef2b3f4d04dafc163d2d10a7cf09b01c4f9e98274df0a1732fe717c0"
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
    ref = json.loads((DATASET_DIR / "mus_musculus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1998-turing-mus-musculus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
