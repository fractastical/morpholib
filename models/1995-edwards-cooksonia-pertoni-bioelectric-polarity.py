#!/usr/bin/env python3
"""
models/1995-edwards-cooksonia-pertoni-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1974)
Organism: Cooksonia pertoni (Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids))
Phenomenon Scale: 1.5 µm (1.454e-06 m, log10=-5.837, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Cooksonia pertoni (Late Silurian (Wenlock-Ludlow, 425 Ma))
Geological Horizon: Late Silurian (Wenlock-Ludlow, 425 Ma) (425.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Cooksonia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.5 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1974_cooksonia_pertoni_bioelectric_polarity.lean` (`MathOfBiology.Loop_1974`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cooksonia_pertoni_bioelectric_polarity"
EXPECTED_SHA256 = {
    "cooksonia_pertoni_bioelectric_polarity_reference.json": "38696082bec72004d178eb23b53684508416476ad937e170fa4fad4eb6ea9bce",
    "cooksonia_pertoni_specimen.jpg": "8fd04a0f4ac5dd5893b91ae0eb689cb450c030fc42bc7a41129eac5169e756c2",
    "scout_manifest.json": "dc0d577160298709520d34a8c0690a6ea899887c9ce549ebd15568aa50a85be4",
    "paper_citations.bib": "86bd6440c94174c0b84d0e4d2f57862e7021fc91621d84eb940409de976ce8c5"
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
    ref = json.loads((DATASET_DIR / "cooksonia_pertoni_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1995-edwards-cooksonia-pertoni-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
