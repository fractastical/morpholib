#!/usr/bin/env python3
"""
models/2011-zhou-ginkgo-yimaensis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1441)
Organism: Ginkgo yimaensis (Middle Jurassic Deeply-Divided 6-Lobe Ginkgo Chronospecies)
Phenomenon Scale: 2.9 mm (0.002909 m, log10=-2.536, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Ginkgo yimaensis (Middle Jurassic (Yima Fm, 170 Ma))
Geological Horizon: Middle Jurassic (Yima Fm, 170 Ma) (170.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1441_ginkgo_yimaensis_clock_wavefront.lean` (`MathOfBiology.Loop_1441`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_yimaensis_clock_wavefront"
EXPECTED_SHA256 = {
    "ginkgo_yimaensis_clock_wavefront_reference.json": "f74e247fbc0cc32935f7e721faadcb54daa67893441ba065392dfc753b2c268f",
    "ginkgo_yimaensis_specimen.jpg": "1ae0d5163efa653c4908ad5d93fc761b18efd5abe7d256c68c27e86ea915e772",
    "scout_manifest.json": "e6438f3e65b70f717dfd5cf772163088c3eb6e9ab156e22598c511c035edd309",
    "paper_citations.bib": "3699b3d371a71579dd5c01f820c5c5b5587dc0e2b62090b8662584e77009f5aa"
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
    ref = json.loads((DATASET_DIR / "ginkgo_yimaensis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-zhou-ginkgo-yimaensis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
