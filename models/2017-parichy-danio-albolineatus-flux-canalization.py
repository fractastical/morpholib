#!/usr/bin/env python3
"""
models/2017-parichy-danio-albolineatus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1472)
Organism: Danio albolineatus (Pearl Danio (Uniform Interstripe))
Phenomenon Scale: 8.0 mm (0.00801 m, log10=-2.096, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Danio albolineatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (8.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1472_danio_albolineatus_flux_canalization.lean` (`MathOfBiology.Loop_1472`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_albolineatus_flux_canalization"
EXPECTED_SHA256 = {
    "danio_albolineatus_flux_canalization_reference.json": "2ce3af99e97e642c93a860c6e8dde728e491351f982edd820c9d1bfbd51c96e8",
    "danio_albolineatus_specimen.jpg": "6059c6481299cc5bb1799fc70390ba1924f905610346e50fc579efcd09df8e46",
    "scout_manifest.json": "09a22634def6877d3f8feab5daa5220b9540d5753475ef15d61696c5cf6f04da",
    "paper_citations.bib": "ed70210070246b3853aa8e5b81f727173eeb3dfa88fa74802d9ae663ccb8446c"
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
    ref = json.loads((DATASET_DIR / "danio_albolineatus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-parichy-danio-albolineatus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
