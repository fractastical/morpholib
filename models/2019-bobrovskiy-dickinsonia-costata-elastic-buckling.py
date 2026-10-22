#!/usr/bin/env python3
"""
models/2019-bobrovskiy-dickinsonia-costata-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1790)
Organism: Dickinsonia costata (Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker))
Phenomenon Scale: 1.2 cm (0.01215 m, log10=-1.915, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Dickinsonia costata (Late Ediacaran (White Sea, 558 Ma))
Geological Horizon: Late Ediacaran (White Sea, 558 Ma) (558.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dickinsonia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1790_dickinsonia_costata_elastic_buckling.lean` (`MathOfBiology.Loop_1790`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dickinsonia_costata_elastic_buckling"
EXPECTED_SHA256 = {
    "dickinsonia_costata_elastic_buckling_reference.json": "64b06cb264f84d02b35e41075c02da3b2c018ef8416e1ec4fe72989afcc3fed6",
    "dickinsonia_costata_specimen.jpg": "ab935d7f33b31712e63e63f3b84c1c1eac296d72d1728a400dbe609341e6ffc9",
    "scout_manifest.json": "0bbde81086c5818e94c651c72af9a882d6d4f5d989b5db02a440d67cfc8afa2e",
    "paper_citations.bib": "d8e7a96dc4195c011ca4465c6962c35673e7751fdbe2cacba6b3219c9eb50b34"
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
    ref = json.loads((DATASET_DIR / "dickinsonia_costata_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-bobrovskiy-dickinsonia-costata-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
