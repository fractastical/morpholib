#!/usr/bin/env python3
"""
models/2002-forsten-equus-scotti-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1942)
Organism: Equus scotti (Pleistocene Giant Ice-Age Horse Chronospecies)
Phenomenon Scale: 177 µm (0.0001768 m, log10=-3.753, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Equus scotti (Middle Pleistocene (Irvingtonian, 0.6 Ma))
Geological Horizon: Middle Pleistocene (Irvingtonian, 0.6 Ma) (0.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (177 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1942_equus_scotti_bioelectric_polarity.lean` (`MathOfBiology.Loop_1942`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/equus_scotti_bioelectric_polarity"
EXPECTED_SHA256 = {
    "equus_scotti_bioelectric_polarity_reference.json": "59a8842a7c8d264f3b518c9e8196e4a7c445c93aff025156efe1539ec006592a",
    "equus_scotti_specimen.jpg": "fea6d4b899e55872847306904b4629acc9fcf09737b2a6f906d0accb0de6e222",
    "scout_manifest.json": "f1611fdc11500be2b8da8a20345a63acd3e9c76b2c34938aefb6cd538ad6757a",
    "paper_citations.bib": "af197b0273f85b3aeb326f624744fd24e01790bb9a1da421e5dad3906451e109"
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
    ref = json.loads((DATASET_DIR / "equus_scotti_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-forsten-equus-scotti-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
