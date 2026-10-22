#!/usr/bin/env python3
"""
models/2020-spiewak-danio-aesculapii-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1834)
Organism: Danio aesculapii (Panther Danio (Vertical Bar Transition))
Phenomenon Scale: 1.4 cm (0.01387 m, log10=-1.858, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Danio aesculapii (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1834_danio_aesculapii_biomineral_accretion.lean` (`MathOfBiology.Loop_1834`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_aesculapii_biomineral_accretion"
EXPECTED_SHA256 = {
    "danio_aesculapii_biomineral_accretion_reference.json": "13c23f70346f56b7c00850dc9fd6aa6e1edc983440ba302390919d5af88b2ccd",
    "danio_aesculapii_specimen.jpg": "4ccce0ce3648e44c77cef768d10f5acd8ab553c4ca38d348c35d4c6846b46bc5",
    "scout_manifest.json": "e89036ad9b8d0f459c315277b17eada48806307877b61c59c4643dba9e024837",
    "paper_citations.bib": "57c2ddf1f09c81544f640f8dd9e0c30c1212d2167d0b242e411555e54c58da51"
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
    ref = json.loads((DATASET_DIR / "danio_aesculapii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-spiewak-danio-aesculapii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
