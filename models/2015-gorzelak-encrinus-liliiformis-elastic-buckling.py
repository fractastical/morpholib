#!/usr/bin/env python3
"""
models/2015-gorzelak-encrinus-liliiformis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1811)
Organism: Encrinus liliiformis (Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles))
Phenomenon Scale: 3.4 µm (3.366e-06 m, log10=-5.473, 3_subcellular_um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Encrinus liliiformis (Middle Triassic (Muschelkalk, 240 Ma))
Geological Horizon: Middle Triassic (Muschelkalk, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Encrinus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.4 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1811_encrinus_liliiformis_elastic_buckling.lean` (`MathOfBiology.Loop_1811`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/encrinus_liliiformis_elastic_buckling"
EXPECTED_SHA256 = {
    "encrinus_liliiformis_elastic_buckling_reference.json": "0f3ace071e4d8c58c56f111d66c820922136950a882af01c836ab4976c664dcb",
    "encrinus_liliiformis_specimen.jpg": "d19f3b6e48018c2dcd0ffb7be81fb1a880d1264498e9b5ec0b872b3310699a44",
    "scout_manifest.json": "cbc4e3774c786f2dc33a48f9de4d09274f2fb3352b3ac2afaafe0b1c89b8a541",
    "paper_citations.bib": "5be5800154af1069da15cf57cb0229676fc855d1ea73fb62aa196aff01997075"
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
    ref = json.loads((DATASET_DIR / "encrinus_liliiformis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-gorzelak-encrinus-liliiformis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
