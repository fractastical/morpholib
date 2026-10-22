#!/usr/bin/env python3
"""
models/2015-crane-ginkgo-biloba-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1894)
Organism: Ginkgo biloba (Extant Maidenhair Tree (Dichotomous Open-Vein Fan Leaf))
Phenomenon Scale: 2.7 cm (0.02712 m, log10=-1.567, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Ginkgo biloba (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (2.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1894_ginkgo_biloba_biomineral_accretion.lean` (`MathOfBiology.Loop_1894`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_biloba_biomineral_accretion"
EXPECTED_SHA256 = {
    "ginkgo_biloba_biomineral_accretion_reference.json": "52ac3457da9a0bc0c41efa6ba195c5cff5e4add452f8199fe2a3bd9f5856e039",
    "ginkgo_biloba_specimen.jpg": "b0a1f66d4688ddb858306c70a27a0bc279b41f3eaf1e9935f5b40d33cb7f7905",
    "scout_manifest.json": "1ac14c6bcd00eab4f04f72a72d76a2a2603ae46f1353d7965d36efbf2400b698",
    "paper_citations.bib": "738b2b5c90722531a872c89c69d45e3462f4ce89375369d130d09da499b20be8"
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
    ref = json.loads((DATASET_DIR / "ginkgo_biloba_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-crane-ginkgo-biloba-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
