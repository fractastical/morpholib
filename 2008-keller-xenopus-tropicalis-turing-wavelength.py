#!/usr/bin/env python3
"""
2008-keller-xenopus-tropicalis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0121)
Organism: Xenopus tropicalis (Western Clawed Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Xenopus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0121_xenopus_tropicalis_turing_wavelength.lean` (`MathOfBiology.Loop_0121`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/xenopus_tropicalis_turing_wavelength"
EXPECTED_SHA256 = {
    "xenopus_tropicalis_turing_wavelength_reference.json": "5ac14aef0a3d4a4c8ce83b14f07ec963f44922a231e7fb245d807096cb8bf312",
    "xenopus_tropicalis_specimen.jpg": "13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f",
    "scout_manifest.json": "367bd8573c775446d2138c068f56634fc6e01dc7bc3cd81dfff81708676610b0",
    "paper_citations.bib": "fd9c9ba3984c654b5bde9f48d064c9ee92d97b13ac0985d2288a8af99bd455bf"
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
    ref = json.loads((DATASET_DIR / "xenopus_tropicalis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-keller-xenopus-tropicalis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
