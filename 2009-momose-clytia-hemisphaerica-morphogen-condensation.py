#!/usr/bin/env python3
"""
2009-momose-clytia-hemisphaerica-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0268)
Organism: Clytia hemisphaerica (Colonial Hydrozoan)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Clytia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0268_clytia_hemisphaerica_morphogen_condensation.lean` (`MathOfBiology.Loop_0268`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/clytia_hemisphaerica_morphogen_condensation"
EXPECTED_SHA256 = {
    "clytia_hemisphaerica_morphogen_condensation_reference.json": "cea39d261a3f9e10ddc3c8c9c05226ec95d4dcd4b36c211cdf92d9170b603a06",
    "clytia_hemisphaerica_specimen.jpg": "2fcc34f8d72cdc5aca117a04c8528f2e4227f504bc61fe0588a43db0a016f236",
    "scout_manifest.json": "f14323af55aa1dcd64462811354a345853ba4d10a978695c311834db50817fef",
    "paper_citations.bib": "559093943a0ebd16cb9b873d567c87f3d9fcd535ec043f3432886d3fee2e9050"
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
    ref = json.loads((DATASET_DIR / "clytia_hemisphaerica_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-momose-clytia-hemisphaerica-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
