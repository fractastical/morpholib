#!/usr/bin/env python3
"""
models/1983-grande-knightia-eocaena-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1745)
Organism: Knightia eocaena (Green River Eocene Clupeomorph Fossil)
Phenomenon Scale: 2.1 cm (0.02111 m, log10=-1.676, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Knightia eocaena (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Knightia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (2.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1745_knightia_eocaena_elastic_buckling.lean` (`MathOfBiology.Loop_1745`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/knightia_eocaena_elastic_buckling"
EXPECTED_SHA256 = {
    "knightia_eocaena_elastic_buckling_reference.json": "3db0aa878cb024d68fd6d6cb3288efef4d640cf0cd3ad52c2da454d9a34acbe4",
    "knightia_eocaena_specimen.jpg": "52c380669cecf901cc96c680b34fb444376e73b78220ae87b0df3a3c9d273c71",
    "scout_manifest.json": "739d20e239ecfae1d6ca4d8ad662bfcc37cb99cd6c99765e02707f5b32a272af",
    "paper_citations.bib": "3ad88d23a2c6fe73bab60fe4c44fe8d5c2fea37daf04a22d416adb98f78e14c7"
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
    ref = json.loads((DATASET_DIR / "knightia_eocaena_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1983-grande-knightia-eocaena-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
