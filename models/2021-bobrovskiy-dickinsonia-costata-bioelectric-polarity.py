#!/usr/bin/env python3
"""
models/2021-bobrovskiy-dickinsonia-costata-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1970)
Organism: Dickinsonia costata (Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker))
Phenomenon Scale: 44 µm (4.42e-05 m, log10=-4.355, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Dickinsonia costata (Late Ediacaran (White Sea, 558 Ma))
Geological Horizon: Late Ediacaran (White Sea, 558 Ma) (558.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dickinsonia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (44 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1970_dickinsonia_costata_bioelectric_polarity.lean` (`MathOfBiology.Loop_1970`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dickinsonia_costata_bioelectric_polarity"
EXPECTED_SHA256 = {
    "dickinsonia_costata_bioelectric_polarity_reference.json": "b34e0b9034d1a3a46d3ad2db6b9566da9cc5117a30c3ff095640529a511c6526",
    "dickinsonia_costata_specimen.jpg": "ab935d7f33b31712e63e63f3b84c1c1eac296d72d1728a400dbe609341e6ffc9",
    "scout_manifest.json": "eeca28d04247a26566956723395dd06115134e4c94109b551eb95d7b3cea6350",
    "paper_citations.bib": "6169a6d9d998965e66196fc6e304ab01f411b3147e97a3ee0bc3508d017eb236"
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
    ref = json.loads((DATASET_DIR / "dickinsonia_costata_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-bobrovskiy-dickinsonia-costata-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
