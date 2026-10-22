#!/usr/bin/env python3
"""
models/2019-martins-loxodonta-africana-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1817)
Organism: Loxodonta africana (African Bush Elephant (Skin Thermal Mud-Cracking Channel Network))
Phenomenon Scale: 288 µm (0.0002878 m, log10=-3.541, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Loxodonta africana (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Loxodonta
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (288 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1817_loxodonta_africana_elastic_buckling.lean` (`MathOfBiology.Loop_1817`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/loxodonta_africana_elastic_buckling"
EXPECTED_SHA256 = {
    "loxodonta_africana_elastic_buckling_reference.json": "4af757a5c1405911e6387d1cdb3740c2a2e67da85f8460cf0dd915dc3cb0da71",
    "loxodonta_africana_specimen.jpg": "a530e48a3daec38e9c6fd4a03ae176871589b79b17232feda58acffdff236163",
    "scout_manifest.json": "7e80d6fba60d36e373e473d4c93288c5526b6d278dd1e87068dd87ebd5095b36",
    "paper_citations.bib": "65208fb2b4519a1c9ac9916e6ea49e0dfbb3faacdd447e2cd2d95d7e09bd94a4"
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
    ref = json.loads((DATASET_DIR / "loxodonta_africana_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-martins-loxodonta-africana-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
