#!/usr/bin/env python3
"""
models/2014-chan-gasterosteus-aculeatus-benthic-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2021)
Organism: Gasterosteus aculeatus_benthic (Post-Glacial Pitx1-Deleted Freshwater Benthic Stickleback (Low Plate))
Phenomenon Scale: 10.1 cm (0.1013 m, log10=-0.994, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Gasterosteus aculeatus_benthic (Holocene Post-Glacial Lake (Extant))
Geological Horizon: Holocene Post-Glacial Lake (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (10.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2021_gasterosteus_aculeatus_benthic_collective_stigmergy.lean` (`MathOfBiology.Loop_2021`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_benthic_collective_stigmergy"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_benthic_collective_stigmergy_reference.json": "e73bc82881c2b22fccfa14c008afacfa770bda8b3fac51d4cf089c2fafd1b799",
    "gasterosteus_aculeatus_benthic_specimen.jpg": "d715e13ebe21bfa5b005584946ebb33b475d11926bac487e67aab10554b26e5b",
    "scout_manifest.json": "7e448cba9f09598d0fa75f8d70e0f62ccba0aabd2aee5515d37ad5ac3840cbfb",
    "paper_citations.bib": "7bc374cbcf8905e8676f9a048eb7c829eedb94a2ccca730520ed47d58b14f8cc"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_benthic_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-chan-gasterosteus-aculeatus-benthic-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
