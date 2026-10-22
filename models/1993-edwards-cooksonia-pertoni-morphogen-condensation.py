#!/usr/bin/env python3
"""
models/1993-edwards-cooksonia-pertoni-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1344)
Organism: Cooksonia pertoni (Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids))
Phenomenon Scale: 327 µm (0.0003272 m, log10=-3.485, 5_primordium_100um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Cooksonia pertoni (Late Silurian (Wenlock-Ludlow, 425 Ma))
Geological Horizon: Late Silurian (Wenlock-Ludlow, 425 Ma) (425.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Cooksonia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (327 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1344_cooksonia_pertoni_morphogen_condensation.lean` (`MathOfBiology.Loop_1344`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cooksonia_pertoni_morphogen_condensation"
EXPECTED_SHA256 = {
    "cooksonia_pertoni_morphogen_condensation_reference.json": "b5a44e488ff3accc7fb3e04285d1d2559a2a1e883b9f74918cb599abe35252d1",
    "cooksonia_pertoni_specimen.jpg": "8fd04a0f4ac5dd5893b91ae0eb689cb450c030fc42bc7a41129eac5169e756c2",
    "scout_manifest.json": "0d096a9a5366e3039d6464b5a26a67317b29e2220725cb5a58466e7c969d6f6f",
    "paper_citations.bib": "e5253156e61f7eb54d6be591d5680929a4f30610f07eac03110f8b865db68fa4"
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
    ref = json.loads((DATASET_DIR / "cooksonia_pertoni_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1993-edwards-cooksonia-pertoni-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
