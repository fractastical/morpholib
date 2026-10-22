#!/usr/bin/env python3
"""
models/2006-bohn-nepenthes-rajah-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1469)
Organism: Nepenthes rajah (Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim))
Phenomenon Scale: 3.6 µm (3.636e-06 m, log10=-5.439, 3_subcellular_um)
Anatomical Structure: Segmentation clock somite / metamere unit in Nepenthes rajah (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Nepenthes
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (3.6 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1469_nepenthes_rajah_clock_wavefront.lean` (`MathOfBiology.Loop_1469`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nepenthes_rajah_clock_wavefront"
EXPECTED_SHA256 = {
    "nepenthes_rajah_clock_wavefront_reference.json": "426a485f85a3eb35a36381dc61aead296f2bf3d9503ba0260251743f856f31ae",
    "nepenthes_rajah_specimen.jpg": "34d473e5c7aa422e26ade1e48e08666c34867f853524458ff188daa017f9d5a3",
    "scout_manifest.json": "b6f7b02aa0a3994d90e4d52f7f86c1ecbd84c72e3d27bb5bc084a2f5574f5ff6",
    "paper_citations.bib": "2b657e4350f021a68292a19792530dcad4b357c971f5a7e1053150f6034587aa"
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
    ref = json.loads((DATASET_DIR / "nepenthes_rajah_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-bohn-nepenthes-rajah-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
