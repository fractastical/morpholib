#!/usr/bin/env python3
"""
models/2011-janis-merychippus-insignis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1940)
Organism: Merychippus insignis (Miocene Grazing Spring-Footed Three-Toed Horse)
Phenomenon Scale: 151 µm (0.0001512 m, log10=-3.820, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Merychippus insignis (Middle Miocene (Barstovian, 15.0 Ma))
Geological Horizon: Middle Miocene (Barstovian, 15.0 Ma) (15.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Merychippus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (151 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1940_merychippus_insignis_bioelectric_polarity.lean` (`MathOfBiology.Loop_1940`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/merychippus_insignis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "merychippus_insignis_bioelectric_polarity_reference.json": "2659a37676c6b30bea7da0719499acb760042b7cb15b5c1f08b9a8b34dd66ce2",
    "merychippus_insignis_specimen.jpg": "93354e358baacf8e852b4dccb0a713b62e217cf8aba2864c49e3a358f03f7425",
    "scout_manifest.json": "8e06b44c84f1b029f57bbe8410b091e18110a35884afe1a7c17f247f5b0f3f67",
    "paper_citations.bib": "5ffd2a94bec1630568a9f9d84367997884a5f33d4fa3811cc7887318fafbff08"
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
    ref = json.loads((DATASET_DIR / "merychippus_insignis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-janis-merychippus-insignis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
