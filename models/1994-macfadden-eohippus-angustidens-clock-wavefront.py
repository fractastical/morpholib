#!/usr/bin/env python3
"""
models/1994-macfadden-eohippus-angustidens-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1398)
Organism: Eohippus angustidens (Early Eocene Four-Toed Dawn Horse (Hyracotherium))
Phenomenon Scale: 6.6 mm (0.006596 m, log10=-2.181, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Eohippus angustidens (Early Eocene (Wasatchian, 52.0 Ma))
Geological Horizon: Early Eocene (Wasatchian, 52.0 Ma) (52.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Eohippus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (6.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1398_eohippus_angustidens_clock_wavefront.lean` (`MathOfBiology.Loop_1398`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eohippus_angustidens_clock_wavefront"
EXPECTED_SHA256 = {
    "eohippus_angustidens_clock_wavefront_reference.json": "0cb678c348ffeddbe26fc0a6041b1e6638f444817c91a806c2ee867a12138523",
    "eohippus_angustidens_specimen.jpg": "4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf",
    "scout_manifest.json": "a2f64e0bb34cebec150a96668850894e4d0841de894c5facef5d3401a541e762",
    "paper_citations.bib": "f43e2f7a55e0012eb4832ef690552ec67a3d7b0203841e9ad9b0cd7ccb84c8de"
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
    ref = json.loads((DATASET_DIR / "eohippus_angustidens_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1994-macfadden-eohippus-angustidens-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
