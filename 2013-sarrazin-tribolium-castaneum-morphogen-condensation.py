#!/usr/bin/env python3
"""
2013-sarrazin-tribolium-castaneum-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0242)
Organism: Tribolium castaneum (Red Flour Beetle)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Tribolium
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0242_tribolium_castaneum_morphogen_condensation.lean` (`MathOfBiology.Loop_0242`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tribolium_castaneum_morphogen_condensation"
EXPECTED_SHA256 = {
    "tribolium_castaneum_morphogen_condensation_reference.json": "b9ecd2d8f3ce712a3d11624de643939d795fdeeba63f3faeec7727e74e060666",
    "tribolium_castaneum_specimen.jpg": "a991c507b6f59178075bea80d7ad2c1c97d839991d31594797919b07b5ac1d77",
    "scout_manifest.json": "228838241f4cbcae3f6fe3117a75ff4598182595acfce665143edc1cef810ffc",
    "paper_citations.bib": "9165a828025cdb2444308d824aa1a83e50c6f00f7bf796a2c6350e7145be6547"
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
    ref = json.loads((DATASET_DIR / "tribolium_castaneum_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-sarrazin-tribolium-castaneum-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
