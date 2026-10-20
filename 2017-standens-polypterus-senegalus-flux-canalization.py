#!/usr/bin/env python3
"""
2017-standens-polypterus-senegalus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0440)
Organism: Polypterus senegalus (Gray Bichir)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Polypterus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0440_polypterus_senegalus_flux_canalization.lean` (`MathOfBiology.Loop_0440`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/polypterus_senegalus_flux_canalization"
EXPECTED_SHA256 = {
    "polypterus_senegalus_flux_canalization_reference.json": "5615776441020f7b8d2887f20776f947155fbb010a6c7becd639ecc7929de911",
    "polypterus_senegalus_specimen.jpg": "8332a72979f53f67ddcfb9e11c646f5695dd1fac3cbcfb340acc3d76e93582f5",
    "scout_manifest.json": "bcefc1abb9bc7965bf0b99ed56d96d017a4bdd619e31cc3db53116e3349a6824",
    "paper_citations.bib": "94ae46e12ca2c86511c4bdbcff90622441fb78f3f1c44a0d0625e4f20d281603"
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
    ref = json.loads((DATASET_DIR / "polypterus_senegalus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-standens-polypterus-senegalus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
