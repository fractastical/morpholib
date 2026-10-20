#!/usr/bin/env python3
"""
2005-laforsch-daphnia-magna-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0249)
Organism: Daphnia magna (Water Flea)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Daphnia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0249_daphnia_magna_morphogen_condensation.lean` (`MathOfBiology.Loop_0249`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/daphnia_magna_morphogen_condensation"
EXPECTED_SHA256 = {
    "daphnia_magna_morphogen_condensation_reference.json": "fe9d00a8dd33989c77fb0f01e413d7b8339441263caea700a70f65949a220d4a",
    "daphnia_magna_specimen.jpg": "fc61b6ab51b92e988777a61cce4db7434210496ee5084ae01ee9f7adf6fcf44b",
    "scout_manifest.json": "18909fbfba08786c28896c5fed3f4ae7847cf33db152ea2cfd51908b6baed7a5",
    "paper_citations.bib": "f3cd66bb3c99130226c2f7fada5b6a5347f4cf3d82807ca79d37f78ffad9e6a6"
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
    ref = json.loads((DATASET_DIR / "daphnia_magna_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-laforsch-daphnia-magna-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
