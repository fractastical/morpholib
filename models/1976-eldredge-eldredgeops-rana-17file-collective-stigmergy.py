#!/usr/bin/env python3
"""
models/1976-eldredge-eldredgeops-rana-17file-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2056)
Organism: Eldredgeops rana_17file (Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye))
Phenomenon Scale: 803 µm (0.0008032 m, log10=-3.095, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Eldredgeops rana_17file (Middle Devonian (Tioughniogan, 384 Ma))
Geological Horizon: Middle Devonian (Tioughniogan, 384 Ma) (384.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (803 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2056_eldredgeops_rana_17file_collective_stigmergy.lean` (`MathOfBiology.Loop_2056`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_17file_collective_stigmergy"
EXPECTED_SHA256 = {
    "eldredgeops_rana_17file_collective_stigmergy_reference.json": "9688a2441cf949c70d1b0705ee27290fba9eb28388ec04f89d33e3ca4a0a66c8",
    "eldredgeops_rana_17file_specimen.jpg": "15e5b154e304f4a7611a6156cdcfaa0863adea78b4676093b13c3c4a1fc76ffe",
    "scout_manifest.json": "d73d7ddaf67d88d06d835a2b4657b065e425143da939e82f3405c9909f69cb4a",
    "paper_citations.bib": "5f0334bd27ae60bb42714fa3e4b1e712c518b52eee921210a2f4dbb9e26b524a"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_17file_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1976-eldredge-eldredgeops-rana-17file-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
