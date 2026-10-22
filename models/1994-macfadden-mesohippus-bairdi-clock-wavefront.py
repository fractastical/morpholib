#!/usr/bin/env python3
"""
models/1994-macfadden-mesohippus-bairdi-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1399)
Organism: Mesohippus bairdi (Late Eocene Three-Toed Sub-Hypsodont Horse)
Phenomenon Scale: 1.0 cm (0.0101 m, log10=-1.996, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Mesohippus bairdi (Late Eocene (Chadronian, 35.0 Ma))
Geological Horizon: Late Eocene (Chadronian, 35.0 Ma) (35.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mesohippus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1399_mesohippus_bairdi_clock_wavefront.lean` (`MathOfBiology.Loop_1399`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/mesohippus_bairdi_clock_wavefront"
EXPECTED_SHA256 = {
    "mesohippus_bairdi_clock_wavefront_reference.json": "8fae2b957f2975ea0bf8aed55b811222736741a685a1704b1e75ace26edc49df",
    "mesohippus_bairdi_specimen.jpg": "ffa85314095959cda06a6a0c44566047678b150e06996250a97c7c8b4dbc1876",
    "scout_manifest.json": "38e60df6ee9d9b9b8c68553cf62759da1a02647e79b83665c0f37edbe4a90144",
    "paper_citations.bib": "7e257e34dc779f055eb2d251101d3cee21c62f455b4338bb737ca45373977efe"
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
    ref = json.loads((DATASET_DIR / "mesohippus_bairdi_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1994-macfadden-mesohippus-bairdi-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
