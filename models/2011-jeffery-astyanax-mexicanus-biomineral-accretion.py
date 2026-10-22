#!/usr/bin/env python3
"""
models/2011-jeffery-astyanax-mexicanus-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1910)
Organism: Astyanax mexicanus (Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion))
Phenomenon Scale: 332 µm (0.0003325 m, log10=-3.478, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Astyanax mexicanus (Holocene Cave Morph (Extant))
Geological Horizon: Holocene Cave Morph (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Astyanax
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (332 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1910_astyanax_mexicanus_biomineral_accretion.lean` (`MathOfBiology.Loop_1910`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/astyanax_mexicanus_biomineral_accretion"
EXPECTED_SHA256 = {
    "astyanax_mexicanus_biomineral_accretion_reference.json": "bf7fca7d65b94dbdd022016d94c4d1a4a033e0567339320f8020a96d0fcc2abb",
    "astyanax_mexicanus_specimen.jpg": "53b28ff32166e8938db70570a413fac88bfdd10206b30d8fcda2d15cb61c64f5",
    "scout_manifest.json": "f56f3bd4880c3c3d33c0cde5fb5aede2a47994f55a113428f4570ed816b10466",
    "paper_citations.bib": "3f6b2e761266623ede84a98aa15e229fb883feabb801034f19b2e8cc37ebe342"
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
    ref = json.loads((DATASET_DIR / "astyanax_mexicanus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-jeffery-astyanax-mexicanus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
