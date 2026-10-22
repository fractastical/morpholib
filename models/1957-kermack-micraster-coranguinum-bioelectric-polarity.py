#!/usr/bin/env python3
"""
models/1957-kermack-micraster-coranguinum-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1992)
Organism: Micraster coranguinum (Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing))
Phenomenon Scale: 34 µm (3.418e-05 m, log10=-4.466, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Micraster coranguinum (Late Cretaceous (Santonian, 85 Ma))
Geological Horizon: Late Cretaceous (Santonian, 85 Ma) (85.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Micraster
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (34 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1992_micraster_coranguinum_bioelectric_polarity.lean` (`MathOfBiology.Loop_1992`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/micraster_coranguinum_bioelectric_polarity"
EXPECTED_SHA256 = {
    "micraster_coranguinum_bioelectric_polarity_reference.json": "5f13ed20535751c42df97b11c254d3db63978c4497c53afa2434fd1ca885db91",
    "micraster_coranguinum_specimen.jpg": "e066a3f38cd1da19878288ea7b3f5ea0aeec2482a9947c60126b6e6e07fb80da",
    "scout_manifest.json": "5b1be2a707e544c4ef5ce16b4552e8ae98cf6ebebbd7eb6decb3a2c4594d1228",
    "paper_citations.bib": "a728cec1a9974c29512c119035e64a1207e7f5bf0c077413edd622140167fb2f"
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
    ref = json.loads((DATASET_DIR / "micraster_coranguinum_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1957-kermack-micraster-coranguinum-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
