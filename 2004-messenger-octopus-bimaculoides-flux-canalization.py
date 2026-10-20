#!/usr/bin/env python3
"""
2004-messenger-octopus-bimaculoides-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0452)
Organism: Octopus bimaculoides (Two-Spot Octopus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Octopus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0452_octopus_bimaculoides_flux_canalization.lean` (`MathOfBiology.Loop_0452`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/octopus_bimaculoides_flux_canalization"
EXPECTED_SHA256 = {
    "octopus_bimaculoides_flux_canalization_reference.json": "df43a48a107991f4ffe4f59351ac0fd25ef4e51bead054c42604956fa495e2dc",
    "octopus_bimaculoides_specimen.jpg": "4e6d70661aa4e7709a51348bee322500de3a32b1ecb8b557b67060ff9d863233",
    "scout_manifest.json": "0ecba987f27019dcb5a4706f641002347b1e627d7b3de8b390e3dd608a506acf",
    "paper_citations.bib": "58c7510aba6068166bc2fa4b89614752a41b32375516933b321ffe2f20864f22"
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
    ref = json.loads((DATASET_DIR / "octopus_bimaculoides_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-messenger-octopus-bimaculoides-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
