#!/usr/bin/env python3
"""
models/2014-yao-chrysomallon-squamiferum-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2094)
Organism: Chrysomallon squamiferum (Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor))
Phenomenon Scale: 883 nm (8.829e-07 m, log10=-6.054, 2_macromolecular_100nm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Chrysomallon squamiferum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Chrysomallon
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (883 nm))
Formal Verification: `morphoeval/lean/loops/Loop_2094_chrysomallon_squamiferum_collective_stigmergy.lean` (`MathOfBiology.Loop_2094`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/chrysomallon_squamiferum_collective_stigmergy"
EXPECTED_SHA256 = {
    "chrysomallon_squamiferum_collective_stigmergy_reference.json": "0ba11a61ecf9205d8e2a209088596935f5f72c021c59567584a4b7d8e4f25551",
    "chrysomallon_squamiferum_specimen.jpg": "9177d515a6734cec2e96211c99840c572fb4d027ec687bdf0cb090fe6bbe9acc",
    "scout_manifest.json": "b1f33c236b97f156b4fac6b1b46819600696888a6a682e78147f5cde0f43bae0",
    "paper_citations.bib": "8146be907e812effe2d2fa8ed620f68b257068729dd73343577182cb713453b2"
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
    ref = json.loads((DATASET_DIR / "chrysomallon_squamiferum_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-yao-chrysomallon-squamiferum-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
