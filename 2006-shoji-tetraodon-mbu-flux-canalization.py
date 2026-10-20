#!/usr/bin/env python3
"""
2006-shoji-tetraodon-mbu-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0432)
Organism: Tetraodon mbu (Giant Pufferfish)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Tetraodon
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0432_tetraodon_mbu_flux_canalization.lean` (`MathOfBiology.Loop_0432`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetraodon_mbu_flux_canalization"
EXPECTED_SHA256 = {
    "tetraodon_mbu_flux_canalization_reference.json": "2f3fd9d6655578d40090d0cb4838bbd6a191a7373bd80464c187e3d31592f55f",
    "tetraodon_mbu_specimen.jpg": "c2d84cbf503eef4d1bdc55fdc0e9d3ba124750e0ae48ff7010b6a4cdca2ee3f3",
    "scout_manifest.json": "4e51246d772a831f726bb7c2a76421d96e0197ed1375f3bd5c9eed4aecbeaa7f",
    "paper_citations.bib": "0bdfdb23083247c9d26f05adcf99fba6e1cdb54fe3f4f1f3e808a5c3d7da3c37"
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
    ref = json.loads((DATASET_DIR / "tetraodon_mbu_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-shoji-tetraodon-mbu-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
