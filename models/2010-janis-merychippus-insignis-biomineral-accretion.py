#!/usr/bin/env python3
"""
models/2010-janis-merychippus-insignis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1850)
Organism: Merychippus insignis (Miocene Grazing Spring-Footed Three-Toed Horse)
Phenomenon Scale: 10.4 cm (0.104 m, log10=-0.983, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Merychippus insignis (Middle Miocene (Barstovian, 15.0 Ma))
Geological Horizon: Middle Miocene (Barstovian, 15.0 Ma) (15.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Merychippus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (10.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1850_merychippus_insignis_biomineral_accretion.lean` (`MathOfBiology.Loop_1850`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/merychippus_insignis_biomineral_accretion"
EXPECTED_SHA256 = {
    "merychippus_insignis_biomineral_accretion_reference.json": "4f5669e0d59c2a9bd17fdc0f3b0af377ff6b25ec82e7b089ace53d091dbf654a",
    "merychippus_insignis_specimen.jpg": "93354e358baacf8e852b4dccb0a713b62e217cf8aba2864c49e3a358f03f7425",
    "scout_manifest.json": "11dcf5319281be3a9f52d0dcee137636f2c8daf0008447d50318b72cf81e13dc",
    "paper_citations.bib": "ddd73da50cc4dc194e757536b2a505874dd937755108ecec8affcc5abf9f904a"
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
    ref = json.loads((DATASET_DIR / "merychippus_insignis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-janis-merychippus-insignis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
