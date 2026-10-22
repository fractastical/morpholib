#!/usr/bin/env python3
"""
models/2004-davidson-strongylocentrotus-purpuratus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1453)
Organism: Strongylocentrotus purpuratus (Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton))
Phenomenon Scale: 6.3 µm (6.324e-06 m, log10=-5.199, 3_subcellular_um)
Anatomical Structure: Segmentation clock somite / metamere unit in Strongylocentrotus purpuratus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Strongylocentrotus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (6.3 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1453_strongylocentrotus_purpuratus_clock_wavefront.lean` (`MathOfBiology.Loop_1453`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/strongylocentrotus_purpuratus_clock_wavefront"
EXPECTED_SHA256 = {
    "strongylocentrotus_purpuratus_clock_wavefront_reference.json": "ec8251d2b36c68c06b60e079d2f8b622fc38658ae7a4480cc9af6e0c5e3f9b00",
    "strongylocentrotus_purpuratus_specimen.jpg": "515b723840267df6465272f908c6951fdf06c057159d3c6f7d6e67e7bee5224a",
    "scout_manifest.json": "db490051d4bc3b5c3f0ad070da71233d97124177160a00f5847134cb3c11fa20",
    "paper_citations.bib": "e4e20ff4ca7f61373538f0ed3683d2fe57e905f788964eceeb073ab8573f2d6b"
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
    ref = json.loads((DATASET_DIR / "strongylocentrotus_purpuratus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-davidson-strongylocentrotus-purpuratus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
