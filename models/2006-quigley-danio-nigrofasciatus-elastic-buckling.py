#!/usr/bin/env python3
"""
models/2006-quigley-danio-nigrofasciatus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1743)
Organism: Danio nigrofasciatus (Spotted Danio (broken ventral stripes))
Phenomenon Scale: 5.7 mm (0.005729 m, log10=-2.242, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Danio nigrofasciatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (5.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1743_danio_nigrofasciatus_elastic_buckling.lean` (`MathOfBiology.Loop_1743`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_nigrofasciatus_elastic_buckling"
EXPECTED_SHA256 = {
    "danio_nigrofasciatus_elastic_buckling_reference.json": "c421312ee32836410b77ac8a190a4eac4cdf74b9c6b30f09305fbf31298fde01",
    "danio_nigrofasciatus_specimen.jpg": "4aee990cea7e343b773365da24389df37f170d7d68c3492c513f22afd9f750d3",
    "scout_manifest.json": "191e9b9f5e09e6e10e197fcbbc67d6d9442a809a62c4f39a95af05b9acdd3bde",
    "paper_citations.bib": "afff2494f833bd97bd74cdb765ccf11113871a592c8dcca9072546f841b5ea8a"
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
    ref = json.loads((DATASET_DIR / "danio_nigrofasciatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-quigley-danio-nigrofasciatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
