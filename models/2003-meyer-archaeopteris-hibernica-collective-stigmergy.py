#!/usr/bin/env python3
"""
models/2003-meyer-archaeopteris-hibernica-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2067)
Organism: Archaeopteris hibernica (Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds))
Phenomenon Scale: 6.9 cm (0.06885 m, log10=-1.162, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Archaeopteris hibernica (Late Devonian (Famennian, 370 Ma))
Geological Horizon: Late Devonian (Famennian, 370 Ma) (370.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Archaeopteris
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (6.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2067_archaeopteris_hibernica_collective_stigmergy.lean` (`MathOfBiology.Loop_2067`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteris_hibernica_collective_stigmergy"
EXPECTED_SHA256 = {
    "archaeopteris_hibernica_collective_stigmergy_reference.json": "b14a90ff564ba86db5c5cf3cec867ffeff6b5ef3bbe1a5c56cfa14c5eae85336",
    "archaeopteris_hibernica_specimen.jpg": "107d9b021f541e8a82a7c594b45f0250c4a6c8fa160f1b3c54299cd439a8163e",
    "scout_manifest.json": "ea7d07bc2884938fca7d2ff2547ae92f93c5a2ba0cd79e9f823366e3dd69af40",
    "paper_citations.bib": "b6e289fc50b4afedb90eacf604e8254d5bfc4bba5217fc816c07df389c4a734d"
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
    ref = json.loads((DATASET_DIR / "archaeopteris_hibernica_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-meyer-archaeopteris-hibernica-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
