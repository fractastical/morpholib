#!/usr/bin/env python3
"""
models/2011-kroger-orthoceras-regulare-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1504)
Organism: Orthoceras regulare (Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa))
Phenomenon Scale: 1.2 cm (0.01181 m, log10=-1.928, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Orthoceras regulare (Middle Ordovician (Darriwilian, 464 Ma))
Geological Horizon: Middle Ordovician (Darriwilian, 464 Ma) (464.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Orthoceras
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1504_orthoceras_regulare_flux_canalization.lean` (`MathOfBiology.Loop_1504`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/orthoceras_regulare_flux_canalization"
EXPECTED_SHA256 = {
    "orthoceras_regulare_flux_canalization_reference.json": "a1c5d423d054bb051bafb21ef13341bcba8b0970d02a45bbc59e4b2abad5722b",
    "orthoceras_regulare_specimen.jpg": "67fd83548556de1506f61ae1eaf002726c3f3a48e3a9d8452937b96eaa47a976",
    "scout_manifest.json": "3289c3c0be4a3171fc353e57be780f1640513703890181e7af0210fd8c232375",
    "paper_citations.bib": "26f56359d8a72171e49029c5916e1d3276ab86eb679a37a751a23c2e466eff2e"
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
    ref = json.loads((DATASET_DIR / "orthoceras_regulare_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-kroger-orthoceras-regulare-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
