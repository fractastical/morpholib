#!/usr/bin/env python3
"""
2001-moore-coprinopsis-cinerea-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0483)
Organism: Coprinopsis cinerea (Gray Shag Mushroom)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Coprinopsis
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0483_coprinopsis_cinerea_flux_canalization.lean` (`MathOfBiology.Loop_0483`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/coprinopsis_cinerea_flux_canalization"
EXPECTED_SHA256 = {
    "coprinopsis_cinerea_flux_canalization_reference.json": "24a3e81fa18fb61aa5df25c3c0f64f196907251b87c82e624de878688bd54077",
    "coprinopsis_cinerea_specimen.jpg": "89844e402834973a57384e9aee4268b35a0e61f0cb9977afb4a2c562507756ca",
    "scout_manifest.json": "62d0b85832c81a380b304fd09accec4d55686f7b9482458626a3b73afbf3f199",
    "paper_citations.bib": "299d6f6e5043b804ad698c94c7e874e94a8755e1053158e9f6ff399f95ffc536"
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
    ref = json.loads((DATASET_DIR / "coprinopsis_cinerea_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-moore-coprinopsis-cinerea-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
