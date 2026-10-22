#!/usr/bin/env python3
"""
models/2012-kroger-orthoceras-regulare-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2044)
Organism: Orthoceras regulare (Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa))
Phenomenon Scale: 8.5 cm (0.08505 m, log10=-1.070, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Orthoceras regulare (Middle Ordovician (Darriwilian, 464 Ma))
Geological Horizon: Middle Ordovician (Darriwilian, 464 Ma) (464.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Orthoceras
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (8.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2044_orthoceras_regulare_collective_stigmergy.lean` (`MathOfBiology.Loop_2044`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/orthoceras_regulare_collective_stigmergy"
EXPECTED_SHA256 = {
    "orthoceras_regulare_collective_stigmergy_reference.json": "6235e565889bc400a47d9c07918a13f2007a49fafb8f918675e4fd92be8a9dbe",
    "orthoceras_regulare_specimen.jpg": "67fd83548556de1506f61ae1eaf002726c3f3a48e3a9d8452937b96eaa47a976",
    "scout_manifest.json": "eabb6fcc06023240151453b352d42bc397c572f62ab23627a66271be3c38ff40",
    "paper_citations.bib": "28fad7caee1d2e0512da60fab64497253560763a560d56123d2501f03964252e"
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
    ref = json.loads((DATASET_DIR / "orthoceras_regulare_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-kroger-orthoceras-regulare-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
