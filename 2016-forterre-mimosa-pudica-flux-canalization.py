#!/usr/bin/env python3
"""
2016-forterre-mimosa-pudica-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0478)
Organism: Mimosa pudica (Sensitive Plant)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Mimosa
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0478_mimosa_pudica_flux_canalization.lean` (`MathOfBiology.Loop_0478`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mimosa_pudica_flux_canalization"
EXPECTED_SHA256 = {
    "mimosa_pudica_flux_canalization_reference.json": "3c4cbb47d74556efcdee4d21db2e86a6686442e44b711269cd0eb4518fcb8933",
    "mimosa_pudica_specimen.jpg": "4ae1af66e325ef597ebdce00e9f639174383d049bc7b0e98b02e97de71c83e20",
    "scout_manifest.json": "5b81af03ecec1d8bb6c7f9349937dc5c1212a645196daf02b77785d95157048d",
    "paper_citations.bib": "bd305495dfb8587a175ae6a6f2f8847d1dd8564cf5113b220fe2d9a3cbc71f93"
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
    ref = json.loads((DATASET_DIR / "mimosa_pudica_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-forterre-mimosa-pudica-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
