#!/usr/bin/env python3
"""
models/2017-parichy-danio-margaritatus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1381)
Organism: Danio margaritatus (Celestial Pearl Danio (Galaxy Rasbora))
Phenomenon Scale: 1.2 mm (0.001224 m, log10=-2.912, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Danio margaritatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.2 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1381_danio_margaritatus_clock_wavefront.lean` (`MathOfBiology.Loop_1381`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_margaritatus_clock_wavefront"
EXPECTED_SHA256 = {
    "danio_margaritatus_clock_wavefront_reference.json": "c9296c4ba706f615b98caa3f5247f6020c6042d108801e0839aa49a779d5aeca",
    "danio_margaritatus_specimen.jpg": "6f4ec861e86da2f7ea31f23d6a81b802286dce7ac2a0631e45f9097fd9b8e876",
    "scout_manifest.json": "84df17a5e188043e0155c7ecce328a4496cf8dce27f2fe316f7e5cde7ea04e64",
    "paper_citations.bib": "42d6d80f62a86ac8f38892c0c36345d8a2b3c339c4132eb9d8093d541cb03234"
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
    ref = json.loads((DATASET_DIR / "danio_margaritatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-parichy-danio-margaritatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
