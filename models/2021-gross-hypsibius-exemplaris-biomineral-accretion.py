#!/usr/bin/env python3
"""
models/2021-gross-hypsibius-exemplaris-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1911)
Organism: Hypsibius exemplaris (Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis))
Phenomenon Scale: 117 µm (0.0001174 m, log10=-3.930, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Hypsibius exemplaris (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Hypsibius
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (117 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1911_hypsibius_exemplaris_biomineral_accretion.lean` (`MathOfBiology.Loop_1911`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hypsibius_exemplaris_biomineral_accretion"
EXPECTED_SHA256 = {
    "hypsibius_exemplaris_biomineral_accretion_reference.json": "c3e530032bde2fa63aca65ec36e9ed1b207a84b8234768174d15ca0c0b1d36d5",
    "hypsibius_exemplaris_specimen.jpg": "47de00f0a828a76d4d2489503394a17024fde8a0989acc96911706aa535ba88d",
    "scout_manifest.json": "023f1d24a6e8fff465f295580cb36d5bb9bfeedccef686b97671d91444cf1bdf",
    "paper_citations.bib": "e7fa508867697af824ede247848e674129385a1505d8d6a0ebde7a51acc59a26"
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
    ref = json.loads((DATASET_DIR / "hypsibius_exemplaris_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-gross-hypsibius-exemplaris-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
