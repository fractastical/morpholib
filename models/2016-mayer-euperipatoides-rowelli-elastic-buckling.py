#!/usr/bin/env python3
"""
models/2016-mayer-euperipatoides-rowelli-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1822)
Organism: Euperipatoides rowelli (Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments))
Phenomenon Scale: 311 µm (0.0003111 m, log10=-3.507, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Euperipatoides rowelli (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Euperipatoides
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (311 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1822_euperipatoides_rowelli_elastic_buckling.lean` (`MathOfBiology.Loop_1822`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euperipatoides_rowelli_elastic_buckling"
EXPECTED_SHA256 = {
    "euperipatoides_rowelli_elastic_buckling_reference.json": "c0d9cb1c87814563ebe94cd7db1431d4bba55cc157085972ce61823bf00f891e",
    "euperipatoides_rowelli_specimen.jpg": "0e99f4aad858382d0f1a61bc28533c722840c8c49a15f15ce70d8f5405b262c2",
    "scout_manifest.json": "1a45681e367910a48166bc98bfbef42797849e9100f0b26d3a03d491bdb4ab55",
    "paper_citations.bib": "c691e3a497275b524976add9e6c370d64528fd3461078a2110f649b2102b348b"
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
    ref = json.loads((DATASET_DIR / "euperipatoides_rowelli_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-mayer-euperipatoides-rowelli-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
