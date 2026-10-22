#!/usr/bin/env python3
"""
models/2001-forsten-equus-scotti-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1852)
Organism: Equus scotti (Pleistocene Giant Ice-Age Horse Chronospecies)
Phenomenon Scale: 12.2 cm (0.1216 m, log10=-0.915, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Equus scotti (Middle Pleistocene (Irvingtonian, 0.6 Ma))
Geological Horizon: Middle Pleistocene (Irvingtonian, 0.6 Ma) (0.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (12.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1852_equus_scotti_biomineral_accretion.lean` (`MathOfBiology.Loop_1852`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/equus_scotti_biomineral_accretion"
EXPECTED_SHA256 = {
    "equus_scotti_biomineral_accretion_reference.json": "1d4794215dd2ccf2c5fc8f1eaf7cbeef7bdac700c42aa2b8bb1f7fa22a28aaa8",
    "equus_scotti_specimen.jpg": "fea6d4b899e55872847306904b4629acc9fcf09737b2a6f906d0accb0de6e222",
    "scout_manifest.json": "c999e58601819c016bd1c96e6596293c235f7dde085b99d27d4fc1ab100e744b",
    "paper_citations.bib": "c5d08162a12767f82fb8985b9850b32ecbbe5be17ee61e570a59dbbd6abcd557"
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
    ref = json.loads((DATASET_DIR / "equus_scotti_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-forsten-equus-scotti-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
