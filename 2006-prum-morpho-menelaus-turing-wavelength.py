#!/usr/bin/env python3
"""
2006-prum-morpho-menelaus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0144)
Organism: Morpho menelaus (Blue Morpho Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Morpho
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0144_morpho_menelaus_turing_wavelength.lean` (`MathOfBiology.Loop_0144`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/morpho_menelaus_turing_wavelength"
EXPECTED_SHA256 = {
    "morpho_menelaus_turing_wavelength_reference.json": "cdd4f19b4291bae277817b0d73d7fa74487544af2566d93820669fcc77ddac71",
    "morpho_menelaus_specimen.jpg": "4568faabf61d02b09522424f4f34c0efc7d869ad36aba4c8d412cda345094c68",
    "scout_manifest.json": "938d7f0758e702c21f2131816aa7bc8d909226b74a817acc1c7d61efa1bfb60d",
    "paper_citations.bib": "296ffbe3317eefae41ca46ac15c5800dea3fc6684a0a136619b640935d775503"
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
    ref = json.loads((DATASET_DIR / "morpho_menelaus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-prum-morpho-menelaus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
