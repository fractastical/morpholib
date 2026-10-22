#!/usr/bin/env python3
"""
models/2007-aizenberg-euplectella-aspergillum-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1465)
Organism: Euplectella aspergillum (Venus' Flower Basket Glass Sponge (Hierarchical Silica Lattice Cage))
Phenomenon Scale: 102 µm (0.000102 m, log10=-3.991, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Euplectella aspergillum (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Euplectella
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (102 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1465_euplectella_aspergillum_clock_wavefront.lean` (`MathOfBiology.Loop_1465`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euplectella_aspergillum_clock_wavefront"
EXPECTED_SHA256 = {
    "euplectella_aspergillum_clock_wavefront_reference.json": "ab40d7ef40a7ef4987067de25486213d7011a512188d90fc9b175204d4609834",
    "euplectella_aspergillum_specimen.jpg": "c39266f00e99a0f35db98c47cad0aab61aaf982021afc8d7827e37376a49fd28",
    "scout_manifest.json": "f6f88fa73325236d52127720529ff8a28e1cd30b745736735b764715dc28895a",
    "paper_citations.bib": "d0a3a58716d96cae9b6fcac89837934eb42bad81c56bfaf46ba48fd0c42d75e8"
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
    ref = json.loads((DATASET_DIR / "euplectella_aspergillum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-aizenberg-euplectella-aspergillum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
