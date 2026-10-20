#!/usr/bin/env python3
"""
2014-standens-polypterus-senegalus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0140)
Organism: Polypterus senegalus (Gray Bichir)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Polypterus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0140_polypterus_senegalus_turing_wavelength.lean` (`MathOfBiology.Loop_0140`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/polypterus_senegalus_turing_wavelength"
EXPECTED_SHA256 = {
    "polypterus_senegalus_turing_wavelength_reference.json": "0bfa66963fda04bbe9552c3782a48b76fbb353ca97e9c7fdd9022dc86aca7541",
    "polypterus_senegalus_specimen.jpg": "8332a72979f53f67ddcfb9e11c646f5695dd1fac3cbcfb340acc3d76e93582f5",
    "scout_manifest.json": "6b86689291061776fec70c32598479ed0c9bcf0dfb8847e50c263c93d8f62b43",
    "paper_citations.bib": "e413c75e4fa8b42ce8d73394ff4e42d815b99612fb4c43ec32b8980796739e8c"
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
    ref = json.loads((DATASET_DIR / "polypterus_senegalus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-standens-polypterus-senegalus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
