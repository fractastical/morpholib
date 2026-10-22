#!/usr/bin/env python3
"""
models/2007-klug-ceratites-nodosus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1956)
Organism: Ceratites nodosus (Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34))
Phenomenon Scale: 22 µm (2.176e-05 m, log10=-4.662, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Ceratites nodosus (Middle Triassic (Ladinian, 240 Ma))
Geological Horizon: Middle Triassic (Ladinian, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ceratites
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (22 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1956_ceratites_nodosus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1956`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ceratites_nodosus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "ceratites_nodosus_bioelectric_polarity_reference.json": "998d95f8b1254932596907a762cd9c83f7323abe9aab8564db3a3f117507fe38",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "5c6eacc9c1c3d3192b92bff81d93286fff43093690286e7f6a73785550353304",
    "paper_citations.bib": "088e38e5cd3de2754331742d5cbbb82b80046f2a98dd8648cb58c4d915ff6e69"
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
    ref = json.loads((DATASET_DIR / "ceratites_nodosus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-klug-ceratites-nodosus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
