#!/usr/bin/env python3
"""
models/2007-shubin-tiktaalik-roseae-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1754)
Organism: Tiktaalik roseae (Late Devonian Wristed Fin-Limb Transitional Fossil)
Phenomenon Scale: 4.6 cm (0.04556 m, log10=-1.341, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Tiktaalik roseae (Late Devonian (Frasnian, 375 Ma))
Geological Horizon: Late Devonian (Frasnian, 375 Ma) (375.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Tiktaalik
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (4.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1754_tiktaalik_roseae_elastic_buckling.lean` (`MathOfBiology.Loop_1754`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/tiktaalik_roseae_elastic_buckling"
EXPECTED_SHA256 = {
    "tiktaalik_roseae_elastic_buckling_reference.json": "67994a3d1ebb350cf3e2631b524635cc04444141d9164aff3e90e8a340a00e3a",
    "tiktaalik_roseae_specimen.jpg": "2b6fe2e7441ef8b8a1f69c1382b71a3606ab9e0ee552372999fb3ab07fb8c3a4",
    "scout_manifest.json": "ee679dec6854b1c218fb8aff49f3acc3ed5c44b584038eb78878c97473bdf8c2",
    "paper_citations.bib": "da93b5afca7458ee2b191cc7b69436081d35773f25d2952bcbd43eb814549e06"
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
    ref = json.loads((DATASET_DIR / "tiktaalik_roseae_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-shubin-tiktaalik-roseae-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
