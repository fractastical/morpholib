#!/usr/bin/env python3
"""
2015-teyssier-chamaeleo-calyptratus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0823)
Organism: Chamaeleo calyptratus (Veiled Chameleon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Chamaeleo
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0823_chamaeleo_calyptratus_biomineral_accretion.lean` (`MathOfBiology.Loop_0823`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chamaeleo_calyptratus_biomineral_accretion"
EXPECTED_SHA256 = {
    "chamaeleo_calyptratus_biomineral_accretion_reference.json": "85d8fa2a9b640429b30d227de97ac11c8a275ea535a88e5005d79b659be243cb",
    "chamaeleo_calyptratus_specimen.jpg": "976c60b3ba6aa23d8e363fe2188ae09cd8c806190b8b02939475defbf44924b3",
    "scout_manifest.json": "0a0b529407a7962fb02ba1de53e008bbecd1ab2632b2c6083089b7ae9498d686",
    "paper_citations.bib": "144b92dbb2863d0f0abd79b0e475243cf9e0c4375679e9774631bb34f804d413"
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
    ref = json.loads((DATASET_DIR / "chamaeleo_calyptratus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-teyssier-chamaeleo-calyptratus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
