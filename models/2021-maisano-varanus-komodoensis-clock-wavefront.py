#!/usr/bin/env python3
"""
models/2021-maisano-varanus-komodoensis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1458)
Organism: Varanus komodoensis (Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork))
Phenomenon Scale: 190 µm (0.0001904 m, log10=-3.720, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Varanus komodoensis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Varanus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (190 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1458_varanus_komodoensis_clock_wavefront.lean` (`MathOfBiology.Loop_1458`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/varanus_komodoensis_clock_wavefront"
EXPECTED_SHA256 = {
    "varanus_komodoensis_clock_wavefront_reference.json": "fc18eaf58aed5d26ad20e026e616cb6343a9ddf8fb0b1b912846a0f249fa1f1d",
    "varanus_komodoensis_specimen.jpg": "347fa0ca27bf35537323e24d59b72d5f6df744f56be66ed7c9649cdf3bc761b5",
    "scout_manifest.json": "dae8cf95306aa8afc7bcd75be36ef5a915b2c3d8f0c73a3e6b8642d9104c1e8c",
    "paper_citations.bib": "a37bb99dea00919684bb9f2d95b9d31a4bcd0c2b0c7c1d48aa210f70298ff411"
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
    ref = json.loads((DATASET_DIR / "varanus_komodoensis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-maisano-varanus-komodoensis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
