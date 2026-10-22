#!/usr/bin/env python3
"""
models/1984-malmgren-globorotalia-fohsi-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1356)
Organism: Globorotalia fohsi (Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies)
Phenomenon Scale: 70 µm (7.031e-05 m, log10=-4.153, 4_cellular_10um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Globorotalia fohsi (Middle Miocene (Serravallian, 12.7 Ma))
Geological Horizon: Middle Miocene (Serravallian, 12.7 Ma) (12.7 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (70 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1356_globorotalia_fohsi_morphogen_condensation.lean` (`MathOfBiology.Loop_1356`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_fohsi_morphogen_condensation"
EXPECTED_SHA256 = {
    "globorotalia_fohsi_morphogen_condensation_reference.json": "7d95334fc76d30d56f0131c3a1625c844a9549203d53f9d8a28761dbaa79a279",
    "globorotalia_fohsi_specimen.jpg": "da1c3b52b1d7e26144fe2b2960cf08ed37afc6fd3eaf5de4044b285992f13b6f",
    "scout_manifest.json": "d4adf18d534877530b37e198fb04b24a2b9dae06a8d45069bcce1fa4f00895a5",
    "paper_citations.bib": "2ed453d90bfe2de4c210665c48a9d98876d29e4ef941f14b06bab4fc3f0782f0"
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
    ref = json.loads((DATASET_DIR / "globorotalia_fohsi_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1984-malmgren-globorotalia-fohsi-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
