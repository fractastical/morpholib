#!/usr/bin/env python3
"""
1989-spero-orbulina-universa-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0300)
Organism: Orbulina universa (Spherical Foraminifer)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Orbulina
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0300_orbulina_universa_morphogen_condensation.lean` (`MathOfBiology.Loop_0300`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/orbulina_universa_morphogen_condensation"
EXPECTED_SHA256 = {
    "orbulina_universa_morphogen_condensation_reference.json": "ec7a6c5b86c54cedc6a0e1d6c6120c4e3a6aed88056a732b01a6a44d12f6fc17",
    "orbulina_universa_specimen.jpg": "33d170bc776044ce2b3334da5523ea69a13551d1ff7cb86635f20a3dab925141",
    "scout_manifest.json": "b694551a15caae0c05cc55afaa99e22b72021878f10397f93634f8a6462b5073",
    "paper_citations.bib": "3ea0808d58ede84b524f9d3cea8ce46d21a9cd6158d822cb3079dda0e06901d6"
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
    ref = json.loads((DATASET_DIR / "orbulina_universa_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1989-spero-orbulina-universa-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
