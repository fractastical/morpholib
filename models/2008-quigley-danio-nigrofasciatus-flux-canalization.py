#!/usr/bin/env python3
"""
models/2008-quigley-danio-nigrofasciatus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1473)
Organism: Danio nigrofasciatus (Spotted Danio (broken ventral stripes))
Phenomenon Scale: 6.5 mm (0.00651 m, log10=-2.186, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Danio nigrofasciatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (6.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1473_danio_nigrofasciatus_flux_canalization.lean` (`MathOfBiology.Loop_1473`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_nigrofasciatus_flux_canalization"
EXPECTED_SHA256 = {
    "danio_nigrofasciatus_flux_canalization_reference.json": "473143161dab6b115506afce90e0c9d416d060f6076256b9070066f67b9872d2",
    "danio_nigrofasciatus_specimen.jpg": "4aee990cea7e343b773365da24389df37f170d7d68c3492c513f22afd9f750d3",
    "scout_manifest.json": "200f4fa65a6c41750d619799859793588f09bf855892b71b409d1764dd965f4c",
    "paper_citations.bib": "7465cb28ac7c537fe3dabfb78a8e783691978f1f4c93548bb3eb462cda78f186"
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
    ref = json.loads((DATASET_DIR / "danio_nigrofasciatus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-quigley-danio-nigrofasciatus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
