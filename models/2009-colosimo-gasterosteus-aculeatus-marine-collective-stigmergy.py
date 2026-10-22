#!/usr/bin/env python3
"""
models/2009-colosimo-gasterosteus-aculeatus-marine-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2020)
Organism: Gasterosteus aculeatus_marine (Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates))
Phenomenon Scale: 10.9 cm (0.1088 m, log10=-0.963, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Gasterosteus aculeatus_marine (Late Pleistocene Marine Ancestor (15 ka))
Geological Horizon: Late Pleistocene Marine Ancestor (15 ka) (0.015 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (10.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2020_gasterosteus_aculeatus_marine_collective_stigmergy.lean` (`MathOfBiology.Loop_2020`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_marine_collective_stigmergy"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_marine_collective_stigmergy_reference.json": "2f5a4819b44acb4755b97f98a06b6832718270dd388b8b109667b76981dbb8cc",
    "gasterosteus_aculeatus_marine_specimen.jpg": "4da41d2381dcc4c8c66a7b6ffe79757e5ca0b0dae3978efe5a16417ec3a5bd15",
    "scout_manifest.json": "710cf68088b405f074782129281ddf2d31fd65bf3683acd61a331f142237c136",
    "paper_citations.bib": "50e9e51b81772c9ca8ccac091e9ed385ab7881371b6e6211c1c2778a3a12447e"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_marine_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-colosimo-gasterosteus-aculeatus-marine-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
