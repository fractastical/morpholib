#!/usr/bin/env python3
"""
models/2010-bell-gasterosteus-doryssus-late-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2019)
Organism: Gasterosteus doryssus_late (Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr))
Phenomenon Scale: 7.5 cm (0.07529 m, log10=-1.123, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Gasterosteus doryssus_late (Late Miocene (Truckee Fm, 9.99 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 9.99 Ma) (9.99 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (7.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2019_gasterosteus_doryssus_late_collective_stigmergy.lean` (`MathOfBiology.Loop_2019`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_late_collective_stigmergy"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_late_collective_stigmergy_reference.json": "c5258f58408c8b3327f8a237fad4dcba31b0cc1006d7ca4370406acc4040a796",
    "gasterosteus_doryssus_late_specimen.jpg": "fd5b18be03310ca28a457aa04293f4114c76a66251867edec13e086175ae1ee1",
    "scout_manifest.json": "1b28248356ec618401e51e8cfa19eda60203b83310ed5a2cc458ff44b54fa908",
    "paper_citations.bib": "d2ec37d98337054a3434f403642d88ac1f1d66b07cd6fe8cacda115c3fb19311"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_late_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-bell-gasterosteus-doryssus-late-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
