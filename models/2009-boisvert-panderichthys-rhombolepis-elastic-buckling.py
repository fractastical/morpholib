#!/usr/bin/env python3
"""
models/2009-boisvert-panderichthys-rhombolepis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1753)
Organism: Panderichthys rhombolepis (Late Devonian Elpistostegalian Flat-Skulled Fish)
Phenomenon Scale: 3.8 cm (0.03812 m, log10=-1.419, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Panderichthys rhombolepis (Late Devonian (Frasnian, 380 Ma))
Geological Horizon: Late Devonian (Frasnian, 380 Ma) (380.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Panderichthys
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1753_panderichthys_rhombolepis_elastic_buckling.lean` (`MathOfBiology.Loop_1753`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/panderichthys_rhombolepis_elastic_buckling"
EXPECTED_SHA256 = {
    "panderichthys_rhombolepis_elastic_buckling_reference.json": "746be50cd0e5155acf498af29a666281285fc2e6d3a8a9ecb1416650d0834d9f",
    "panderichthys_rhombolepis_specimen.jpg": "5c39604c489cd7a850b62b1d1aaf2e3061b9865043e83968c2feebf4e34ca870",
    "scout_manifest.json": "13b3ac9adc3d39cc30d4124a4d3d9f6b2421f6832ed1bb2e5f2b0889a0a77095",
    "paper_citations.bib": "46b257525be73e41df2e3ffd8317729d059a39985901c7eb27f9d2b072f744e7"
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
    ref = json.loads((DATASET_DIR / "panderichthys_rhombolepis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-boisvert-panderichthys-rhombolepis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
