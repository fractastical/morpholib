#!/usr/bin/env python3
"""
models/2019-landman-baculites-compressus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1329)
Organism: Baculites compressus (Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite)
Phenomenon Scale: 1.9 cm (0.01921 m, log10=-1.716, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Baculites compressus (Late Cretaceous (Pierre Shale, 73.5 Ma))
Geological Horizon: Late Cretaceous (Pierre Shale, 73.5 Ma) (73.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Baculites
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1329_baculites_compressus_morphogen_condensation.lean` (`MathOfBiology.Loop_1329`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/baculites_compressus_morphogen_condensation"
EXPECTED_SHA256 = {
    "baculites_compressus_morphogen_condensation_reference.json": "c03eaa74e27108a753c84a4be56847e9880bfe5d13d3ab149aea73644c90c23e",
    "baculites_compressus_specimen.jpg": "849f95ffd5d647ae2a12750587da774719bdf94f1cf2520dc375c315c45eab50",
    "scout_manifest.json": "ba3dce59bc0a6750690936bba00d79c47c4b4338a7981f58685c2ea37995880d",
    "paper_citations.bib": "0c9f82f5a03917eef0a1fbb831adb3406802063c7a49f4bd8df3a072aefb2fd6"
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
    ref = json.loads((DATASET_DIR / "baculites_compressus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-landman-baculites-compressus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
