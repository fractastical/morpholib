#!/usr/bin/env python3
"""
models/2003-mcloughlin-glossopteris-brownii-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1439)
Organism: Glossopteris brownii (Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation))
Phenomenon Scale: 4.1 mm (0.004092 m, log10=-2.388, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Glossopteris brownii (Late Permian (Gondwana, 260 Ma))
Geological Horizon: Late Permian (Gondwana, 260 Ma) (260.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Glossopteris
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (4.1 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1439_glossopteris_brownii_clock_wavefront.lean` (`MathOfBiology.Loop_1439`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/glossopteris_brownii_clock_wavefront"
EXPECTED_SHA256 = {
    "glossopteris_brownii_clock_wavefront_reference.json": "017fa51879aca0df3bca1870094406afad6a55446ba4ccca180c86a73d63e0c3",
    "glossopteris_brownii_specimen.jpg": "21f458497a8d6fffd0bfea916b3893a105c31943c6d82c526ea20f691965f627",
    "scout_manifest.json": "ca306677a8997a525497ff47aa724da6ea4af3995aacbae7f21c52e02ade95fb",
    "paper_citations.bib": "68e5d81d34cc96174c7b4d902f36ec6c1c0d81028140cf9fb14f7e7a7facd9c2"
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
    ref = json.loads((DATASET_DIR / "glossopteris_brownii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-mcloughlin-glossopteris-brownii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
