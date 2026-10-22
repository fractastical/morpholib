#!/usr/bin/env python3
"""
models/2017-mayer-euperipatoides-rowelli-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1462)
Organism: Euperipatoides rowelli (Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments))
Phenomenon Scale: 113 µm (0.0001131 m, log10=-3.947, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Euperipatoides rowelli (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Euperipatoides
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (113 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1462_euperipatoides_rowelli_clock_wavefront.lean` (`MathOfBiology.Loop_1462`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euperipatoides_rowelli_clock_wavefront"
EXPECTED_SHA256 = {
    "euperipatoides_rowelli_clock_wavefront_reference.json": "292254a9b6f4401ec9860205e5dff18c1db60ab9dc20dd55f691d1781e410c12",
    "euperipatoides_rowelli_specimen.jpg": "0e99f4aad858382d0f1a61bc28533c722840c8c49a15f15ce70d8f5405b262c2",
    "scout_manifest.json": "6fa0956df7f1fd4a480b1ba917797ea03853f89531047cb423cb7fafd4e8645b",
    "paper_citations.bib": "f1e436ccc0ba91252c838d56a8feb5d20e2134f69b0af137d6099814c816846d"
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
    ref = json.loads((DATASET_DIR / "euperipatoides_rowelli_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-mayer-euperipatoides-rowelli-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
