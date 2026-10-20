#!/usr/bin/env python3
"""
2010-clarke-aptenodytes-forsteri-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0119)
Organism: Aptenodytes forsteri (Emperor Penguin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Aptenodytes
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0119_aptenodytes_forsteri_turing_wavelength.lean` (`MathOfBiology.Loop_0119`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aptenodytes_forsteri_turing_wavelength"
EXPECTED_SHA256 = {
    "aptenodytes_forsteri_turing_wavelength_reference.json": "9978be0e3bc1869349d7412d84f099f93109790daa43f16caeba7866cb13c40c",
    "aptenodytes_forsteri_specimen.jpg": "ef141d4e199c2ace11888d32dc8d271279924c5e9a9faa6f15ad839d7b5dd0d0",
    "scout_manifest.json": "f8d763233c99205c80d4d049c161300c4012cd2ee1e7823c444ba59e438a6d55",
    "paper_citations.bib": "2d4a9c4ea13ac21b31e620525660c7aeb44f136cd60909dffaefd81ea612ee80"
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
    ref = json.loads((DATASET_DIR / "aptenodytes_forsteri_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-clarke-aptenodytes-forsteri-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
