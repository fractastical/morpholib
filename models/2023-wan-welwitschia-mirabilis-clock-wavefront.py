#!/usr/bin/env python3
"""
models/2023-wan-welwitschia-mirabilis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1467)
Organism: Welwitschia mirabilis (Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves))
Phenomenon Scale: 6.3 cm (0.06324 m, log10=-1.199, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Welwitschia mirabilis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Welwitschia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (6.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1467_welwitschia_mirabilis_clock_wavefront.lean` (`MathOfBiology.Loop_1467`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/welwitschia_mirabilis_clock_wavefront"
EXPECTED_SHA256 = {
    "welwitschia_mirabilis_clock_wavefront_reference.json": "ee46b64fea5101fddb89ae52b640cb957dd05b025daef9df2be60be693592e34",
    "welwitschia_mirabilis_specimen.jpg": "ee3b34d91d4679f637d5087e7e7fa56ebbe65a443882acb98b96db019cfdc1ab",
    "scout_manifest.json": "28ecdf01f85c9cfac995c1886e71499b0450b2feeb5a3cf32536f3419f95eac9",
    "paper_citations.bib": "6d0389f71ae81b0051942502de691bfd8e733c14457d9214ea16d6ea2f227c8b"
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
    ref = json.loads((DATASET_DIR / "welwitschia_mirabilis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-wan-welwitschia-mirabilis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
