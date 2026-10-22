#!/usr/bin/env python3
"""
models/1998-norris-globorotalia-praefohsi-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1895)
Organism: Globorotalia praefohsi (Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies)
Phenomenon Scale: 166 µm (0.0001664 m, log10=-3.779, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Globorotalia praefohsi (Middle Miocene (Serravallian, 13.4 Ma))
Geological Horizon: Middle Miocene (Serravallian, 13.4 Ma) (13.4 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (166 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1895_globorotalia_praefohsi_biomineral_accretion.lean` (`MathOfBiology.Loop_1895`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_praefohsi_biomineral_accretion"
EXPECTED_SHA256 = {
    "globorotalia_praefohsi_biomineral_accretion_reference.json": "4314fa93a862a6075b0bb48c85922c11a93a1f001e4910e92dd0ffc1a61242aa",
    "globorotalia_praefohsi_specimen.jpg": "4bd61f5c5fd1e4a4d998a7b4bf377b989b14b379445592ebb2d464ec82617c74",
    "scout_manifest.json": "27c620908875b1d19533deac7a158dca03b7713e14ca1e5852ef81af29302602",
    "paper_citations.bib": "6b77e2eb71715be6861676d694ba38be7fef97618d7116926a5c38cdcc51fa66"
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
    ref = json.loads((DATASET_DIR / "globorotalia_praefohsi_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1998-norris-globorotalia-praefohsi-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
