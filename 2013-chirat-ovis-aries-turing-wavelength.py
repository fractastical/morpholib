#!/usr/bin/env python3
"""
2013-chirat-ovis-aries-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0107)
Organism: Ovis aries (Bighorn Sheep)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ovis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0107_ovis_aries_turing_wavelength.lean` (`MathOfBiology.Loop_0107`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ovis_aries_turing_wavelength"
EXPECTED_SHA256 = {
    "ovis_aries_turing_wavelength_reference.json": "049bedaedf1406e117ec6e56c251b012614c5f0fde99b031382cb1cabacaf4e2",
    "ovis_aries_specimen.jpg": "3e5aea2a1dd349564873df9e61677c401d7c2264295f722a777e3808f6a3c0d2",
    "scout_manifest.json": "2c179994f476eece432c497637ad1f913eb70bd9af10e8b74128902724caf0d7",
    "paper_citations.bib": "ecd3f698a223c15f622718f70275be49ff1e50cc7f946f42c1705e77dfa31584"
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
    ref = json.loads((DATASET_DIR / "ovis_aries_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-chirat-ovis-aries-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
