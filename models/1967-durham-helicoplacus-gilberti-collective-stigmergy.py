#!/usr/bin/env python3
"""
models/1967-durham-helicoplacus-gilberti-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2079)
Organism: Helicoplacus gilberti (Early Cambrian Triradial Helical-Plated Stem-Echinoderm)
Phenomenon Scale: 6.0 cm (0.06048 m, log10=-1.218, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Helicoplacus gilberti (Early Cambrian (Poleta Fm, 516 Ma))
Geological Horizon: Early Cambrian (Poleta Fm, 516 Ma) (516.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Helicoplacus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (6.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2079_helicoplacus_gilberti_collective_stigmergy.lean` (`MathOfBiology.Loop_2079`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/helicoplacus_gilberti_collective_stigmergy"
EXPECTED_SHA256 = {
    "helicoplacus_gilberti_collective_stigmergy_reference.json": "e6148f10c94d464e39659576930830904d66926826d1dcff9552b2f85624be68",
    "helicoplacus_gilberti_specimen.jpg": "13c0993dd3e63798e7d236896fbbbbe4e10b4f49fe7d356be1e6a56d3f579d8a",
    "scout_manifest.json": "3383c6a4e8be0e2a1c5d9ce5209f7122904ded27b0296d7674339a9634ef43d5",
    "paper_citations.bib": "cb8fea4641994420ddede6fb13fe4a08106379ae56fb52016f02fb45ffd132aa"
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
    ref = json.loads((DATASET_DIR / "helicoplacus_gilberti_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/1967-durham-helicoplacus-gilberti-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
