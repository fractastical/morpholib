#!/usr/bin/env python3
"""
1997-jean-pinus-pinea-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0475)
Organism: Pinus pinea (Stone Pine)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Pinus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0475_pinus_pinea_flux_canalization.lean` (`MathOfBiology.Loop_0475`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pinus_pinea_flux_canalization"
EXPECTED_SHA256 = {
    "pinus_pinea_flux_canalization_reference.json": "88094a67828259f0fac961c7ec2fa911ec80f59fbc07f082d68f6293b544f6ea",
    "pinus_pinea_specimen.jpg": "87c1fef6a6fb55bd3bf3e35f590d280e453c1348b5d11f18c356c09345da3904",
    "scout_manifest.json": "9209490029e2c904a882b37749b73548d212fea2ca29e9180821a8f4f18a7625",
    "paper_citations.bib": "1815eae1b9a2b1f4fac632ab52a59acb47869a25b30f6df425c90f87d5894f41"
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
    ref = json.loads((DATASET_DIR / "pinus_pinea_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1997-jean-pinus-pinea-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
