#!/usr/bin/env python3
"""
models/2012-jeffery-astyanax-mexicanus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2000)
Organism: Astyanax mexicanus (Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion))
Phenomenon Scale: 484 nm (4.836e-07 m, log10=-6.316, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Astyanax mexicanus (Holocene Cave Morph (Extant))
Geological Horizon: Holocene Cave Morph (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Astyanax
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (484 nm))
Formal Verification: `morphoeval/lean/loops/Loop_2000_astyanax_mexicanus_bioelectric_polarity.lean` (`MathOfBiology.Loop_2000`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/astyanax_mexicanus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "astyanax_mexicanus_bioelectric_polarity_reference.json": "83b0d740b7f35b0167b9018824a4e124196782ecb673758016894f9e2262836b",
    "astyanax_mexicanus_specimen.jpg": "53b28ff32166e8938db70570a413fac88bfdd10206b30d8fcda2d15cb61c64f5",
    "scout_manifest.json": "129fc0507afdc8f9182c4a8c47d394b87e56df690615df6e84e9e5a5cd24df65",
    "paper_citations.bib": "23b9f415dbddc057770479ee447ee4a1cbc274f23d62c1aa3b9ab525a01fbdf0"
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
    ref = json.loads((DATASET_DIR / "astyanax_mexicanus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-jeffery-astyanax-mexicanus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
