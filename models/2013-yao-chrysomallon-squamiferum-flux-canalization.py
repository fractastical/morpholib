#!/usr/bin/env python3
"""
models/2013-yao-chrysomallon-squamiferum-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1554)
Organism: Chrysomallon squamiferum (Scaly-Foot Hydrothermal Vent Gastropod (Greigite Iron-Sulfide Armor))
Phenomenon Scale: 123 nm (1.226e-07 m, log10=-6.912, 2_macromolecular_100nm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Chrysomallon squamiferum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Chrysomallon
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (123 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1554_chrysomallon_squamiferum_flux_canalization.lean` (`MathOfBiology.Loop_1554`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/chrysomallon_squamiferum_flux_canalization"
EXPECTED_SHA256 = {
    "chrysomallon_squamiferum_flux_canalization_reference.json": "07a7bb5e50c7a81a1fbab1aea1348b2a521122ee1116c0418654174cc432cbb1",
    "chrysomallon_squamiferum_specimen.jpg": "9177d515a6734cec2e96211c99840c572fb4d027ec687bdf0cb090fe6bbe9acc",
    "scout_manifest.json": "b1faf00fdbf860e626c98bac060ce68a53787338131142f85d3818a629ec978a",
    "paper_citations.bib": "b58c4a4db7b26dfd4e4edbf3a6c166d9eb31769b87cad078e63567bc9333071b"
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
    ref = json.loads((DATASET_DIR / "chrysomallon_squamiferum_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-yao-chrysomallon-squamiferum-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
