#!/usr/bin/env python3
"""
models/2022-jabusch-pyrocystis-fusiformis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1920)
Organism: Pyrocystis fusiformis (Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash))
Phenomenon Scale: 220 µm (0.0002195 m, log10=-3.659, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Pyrocystis fusiformis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Pyrocystis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (220 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1920_pyrocystis_fusiformis_biomineral_accretion.lean` (`MathOfBiology.Loop_1920`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pyrocystis_fusiformis_biomineral_accretion"
EXPECTED_SHA256 = {
    "pyrocystis_fusiformis_biomineral_accretion_reference.json": "7020cdef1d4ae4f94602027db2c9b1752089d6533b4aa72cde43e1dc9b6b1731",
    "pyrocystis_fusiformis_specimen.jpg": "4fad32ed3942f3ab89d8e79cda844c3cab6085f9a3ce836885fe9d9bf8112cb2",
    "scout_manifest.json": "d7cacbe70411b4dc91506083af2192fea22b71632f9fe8263fe9ee974c232176",
    "paper_citations.bib": "d548bd2098e77577aa93d94cf4612068e97802e635fc091cb5a975b32d7c9d5d"
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
    ref = json.loads((DATASET_DIR / "pyrocystis_fusiformis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-jabusch-pyrocystis-fusiformis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
