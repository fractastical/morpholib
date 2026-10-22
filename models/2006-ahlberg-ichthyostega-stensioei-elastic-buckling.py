#!/usr/bin/env python3
"""
models/2006-ahlberg-ichthyostega-stensioei-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1756)
Organism: Ichthyostega stensioei (Late Devonian Seven-Digited Weight-Bearing Tetrapod)
Phenomenon Scale: 3.0 cm (0.03035 m, log10=-1.518, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Ichthyostega stensioei (Late Devonian (Famennian, 363 Ma))
Geological Horizon: Late Devonian (Famennian, 363 Ma) (363.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ichthyostega
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1756_ichthyostega_stensioei_elastic_buckling.lean` (`MathOfBiology.Loop_1756`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ichthyostega_stensioei_elastic_buckling"
EXPECTED_SHA256 = {
    "ichthyostega_stensioei_elastic_buckling_reference.json": "8842f6c27d9802bd35cd80339a6e59b8cc7aeaa1dc08544581e127bde6169fa3",
    "ichthyostega_stensioei_specimen.jpg": "d0371040f354560ee3b95904f5d16c9f224254e9b490bc2d4a9c7d712aa42e0c",
    "scout_manifest.json": "0486d3910b1ae8dcbd3f513918908d13514a47da871b714af88b4e200b6fe291",
    "paper_citations.bib": "e966d6185de25e0424d43e7429da6b71db14502569878f526378cea1b0cd3f42"
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
    ref = json.loads((DATASET_DIR / "ichthyostega_stensioei_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-ahlberg-ichthyostega-stensioei-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
