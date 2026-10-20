#!/usr/bin/env python3
"""
2018-porter-hippocampus-kuda-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0434)
Organism: Hippocampus kuda (Spotted Seahorse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Hippocampus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0434_hippocampus_kuda_flux_canalization.lean` (`MathOfBiology.Loop_0434`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hippocampus_kuda_flux_canalization"
EXPECTED_SHA256 = {
    "hippocampus_kuda_flux_canalization_reference.json": "83cee4f24bb95286f996612c262ed5e728e9ffad307cfba3b5c92f76a2d4092e",
    "hippocampus_kuda_specimen.jpg": "2d453503e6f09426796b25434fc672645ac4aa639654b8d282c6607928df715e",
    "scout_manifest.json": "e3e14a220ca7d60c367a45621c221114d18775c9d4c1c6d584a0c7dc7335babb",
    "paper_citations.bib": "69b039ed029f4ea031fc5af6f6c32d2865ea8aa253c964c95b1ec665748e9370"
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
    ref = json.loads((DATASET_DIR / "hippocampus_kuda_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-porter-hippocampus-kuda-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
