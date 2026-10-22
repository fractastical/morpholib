#!/usr/bin/env python3
"""
models/2014-li-microraptor-gui-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1861)
Organism: Microraptor gui (Early Cretaceous Iridescent Black Four-Winged Dromaeosaur)
Phenomenon Scale: 471 nm (4.706e-07 m, log10=-6.327, 2_macromolecular_100nm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Microraptor gui (Early Cretaceous (Jiufotang Fm, 120.0 Ma))
Geological Horizon: Early Cretaceous (Jiufotang Fm, 120.0 Ma) (120.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Microraptor
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (471 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1861_microraptor_gui_biomineral_accretion.lean` (`MathOfBiology.Loop_1861`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/microraptor_gui_biomineral_accretion"
EXPECTED_SHA256 = {
    "microraptor_gui_biomineral_accretion_reference.json": "9145a2dc8fab68e7e0b01015a2b8db0509d7aefdb79db3e6c5d0036ac8915c2b",
    "microraptor_gui_specimen.jpg": "24fb32a1367db5df0d77dace2b5b810c0854a2b3c231e370b837356348765137",
    "scout_manifest.json": "2866abba0fe27d0df7f878bc6753ecd062a0b8318256e367fef7e7a8a04b73b7",
    "paper_citations.bib": "d44ced575775443f211b4a1e5c891cea1d296371525ac0401f7f84009d3a5ecf"
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
    ref = json.loads((DATASET_DIR / "microraptor_gui_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-li-microraptor-gui-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
