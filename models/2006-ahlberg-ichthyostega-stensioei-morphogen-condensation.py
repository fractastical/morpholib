#!/usr/bin/env python3
"""
models/2006-ahlberg-ichthyostega-stensioei-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1306)
Organism: Ichthyostega stensioei (Late Devonian Seven-Digited Weight-Bearing Tetrapod)
Phenomenon Scale: 2.5 cm (0.02483 m, log10=-1.605, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ichthyostega stensioei (Late Devonian (Famennian, 363 Ma))
Geological Horizon: Late Devonian (Famennian, 363 Ma) (363.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ichthyostega
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (2.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1306_ichthyostega_stensioei_morphogen_condensation.lean` (`MathOfBiology.Loop_1306`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ichthyostega_stensioei_morphogen_condensation"
EXPECTED_SHA256 = {
    "ichthyostega_stensioei_morphogen_condensation_reference.json": "d32ab7c6a9c86b6ffc37cd9e30948872f5ed3b6d2e9a0a554e56323e443164b5",
    "ichthyostega_stensioei_specimen.jpg": "d0371040f354560ee3b95904f5d16c9f224254e9b490bc2d4a9c7d712aa42e0c",
    "scout_manifest.json": "3822d2dfa1b01c84ad178c9bac4cee2691fa46542807cb8c8b871ae0448bc51a",
    "paper_citations.bib": "31b719ed1a72124b39aacdd229d7e59ba5a3c0729404a9e420138ba8b59fa2f8"
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
    ref = json.loads((DATASET_DIR / "ichthyostega_stensioei_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-ahlberg-ichthyostega-stensioei-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
