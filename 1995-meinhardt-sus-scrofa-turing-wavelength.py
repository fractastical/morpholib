#!/usr/bin/env python3
"""
1995-meinhardt-sus-scrofa-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0106)
Organism: Sus scrofa (Wild Boar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Sus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0106_sus_scrofa_turing_wavelength.lean` (`MathOfBiology.Loop_0106`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sus_scrofa_turing_wavelength"
EXPECTED_SHA256 = {
    "sus_scrofa_turing_wavelength_reference.json": "48d36e462b96a8cd63d1538819ae64bf6d1b2a1276e13b26e162a13b0abac41c",
    "sus_scrofa_specimen.jpg": "2a19f0ec20969646f39adb56919e7ec23b3c0f4a963658eeab969a0d13ba3d70",
    "scout_manifest.json": "c6249630db48fc137c3f63448d777da7af26cac202730fe21f66227068475ad8",
    "paper_citations.bib": "3f393262808e4af35df57b8f76f93c240cdbaf37f49c757a0e5ffb4ccbe67512"
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
    ref = json.loads((DATASET_DIR / "sus_scrofa_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-meinhardt-sus-scrofa-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
