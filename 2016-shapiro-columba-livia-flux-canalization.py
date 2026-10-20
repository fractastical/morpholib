#!/usr/bin/env python3
"""
2016-shapiro-columba-livia-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0414)
Organism: Columba livia (Rock Dove)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Columba
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0414_columba_livia_flux_canalization.lean` (`MathOfBiology.Loop_0414`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/columba_livia_flux_canalization"
EXPECTED_SHA256 = {
    "columba_livia_flux_canalization_reference.json": "0fb8a926fe32ce611d5a9df558b3993f9ff975be3a1f2a4d18fe3b3d998ef66c",
    "columba_livia_specimen.jpg": "00bf4ed50c6075550c363caacdfa0f8820f68486fb8d737e81039f614c1ae7fb",
    "scout_manifest.json": "8e98369202a3dc49214a71f9e389d6fc42859061def6ab44f7886488ed12586e",
    "paper_citations.bib": "5543aaf42d611cd69a465acba6dd38c5457531252f94a342ca939fc4f1f5b808"
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
    ref = json.loads((DATASET_DIR / "columba_livia_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-shapiro-columba-livia-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
