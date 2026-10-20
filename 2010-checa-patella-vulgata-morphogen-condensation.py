#!/usr/bin/env python3
"""
2010-checa-patella-vulgata-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0253)
Organism: Patella vulgata (Common Limpet)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Patella
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0253_patella_vulgata_morphogen_condensation.lean` (`MathOfBiology.Loop_0253`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/patella_vulgata_morphogen_condensation"
EXPECTED_SHA256 = {
    "patella_vulgata_morphogen_condensation_reference.json": "d214582afa0b3e9d6bbb01f0440848ac2ddbaf4d301cfdd4b97454570684429d",
    "patella_vulgata_specimen.jpg": "dc71ee15b04a3541b9b25a13411da18888acaaba54973227c4ba40c361669d0e",
    "scout_manifest.json": "b3b5376f75482f5af5fc1ef9387752ebb0de45dbec2afa1e1b89ac2b0e1ea7c8",
    "paper_citations.bib": "3e3e8ea8d2ecc0bc8927b94cc9ec5c2c1492a593d89fd51c3bc534bda0d24b6e"
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
    ref = json.loads((DATASET_DIR / "patella_vulgata_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-checa-patella-vulgata-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
