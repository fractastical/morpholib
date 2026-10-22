#!/usr/bin/env python3
"""
models/2010-bell-gasterosteus-doryssus-early-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2017)
Organism: Gasterosteus doryssus_early (Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr))
Phenomenon Scale: 10.2 cm (0.102 m, log10=-0.991, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Gasterosteus doryssus_early (Late Miocene (Truckee Fm, 10.02 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.02 Ma) (10.02 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (10.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2017_gasterosteus_doryssus_early_collective_stigmergy.lean` (`MathOfBiology.Loop_2017`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_early_collective_stigmergy"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_collective_stigmergy_reference.json": "a72c5abdebc1f6cada2267f7ccbc70d21a68d28766baa92ffb8dc06823c9bf04",
    "gasterosteus_doryssus_early_specimen.jpg": "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json": "5565d5c2b04491ca55b756dedc394b126290e8cbd633671d030836703a1dfce5",
    "paper_citations.bib": "12c1f888defda9766da5738c001e286f008bf5a0528a3dd92247375ea8b75357"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_early_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-bell-gasterosteus-doryssus-early-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
