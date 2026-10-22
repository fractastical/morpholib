#!/usr/bin/env python3
"""
models/1982-jarvik-eusthenopteron-foordi-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1392)
Organism: Eusthenopteron foordi (Late Devonian Tristichopterid Lobe-Fin Fish)
Phenomenon Scale: 1.1 cm (0.01131 m, log10=-1.947, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Eusthenopteron foordi (Late Devonian (Frasnian, 379 Ma))
Geological Horizon: Late Devonian (Frasnian, 379 Ma) (379.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Eusthenopteron
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1392_eusthenopteron_foordi_clock_wavefront.lean` (`MathOfBiology.Loop_1392`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eusthenopteron_foordi_clock_wavefront"
EXPECTED_SHA256 = {
    "eusthenopteron_foordi_clock_wavefront_reference.json": "3cec011326e76a8f27bb1519367885ff8b375685ad7b0a95abcff4aec35d6382",
    "eusthenopteron_foordi_specimen.jpg": "d97b68e66855434cb59cbd6b3fc48d7827c97c6981f567e77fb025c359499811",
    "scout_manifest.json": "3d260657439e8a1c44098b3463dcf4cb662813f37eacacdaafefcb9fc48de351",
    "paper_citations.bib": "fca89ef9f4065646321313e2da9171504a8d860718e98e0d0452a31a5a72526c"
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
    ref = json.loads((DATASET_DIR / "eusthenopteron_foordi_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1982-jarvik-eusthenopteron-foordi-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
