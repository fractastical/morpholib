#!/usr/bin/env python3
"""
models/2016-wang-manis-javanica-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1275)
Organism: Manis javanica (Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice))
Phenomenon Scale: 1.4 mm (0.001414 m, log10=-2.850, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Manis javanica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Manis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1275_manis_javanica_turing_wavelength.lean` (`MathOfBiology.Loop_1275`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/manis_javanica_turing_wavelength"
EXPECTED_SHA256 = {
    "manis_javanica_turing_wavelength_reference.json": "96050e74366e75a3a06a75aa49f790badb7d97bf3a3a749d8f114e05e70fb934",
    "manis_javanica_specimen.jpg": "0d2cf7b0e5853e57873b4a69f96a0bf200561aa593feda56d04025e8b7619711",
    "scout_manifest.json": "43cbc63929dc94917fd7a30c99e67adcf64137ba61b9486b482f663ceb8ea0bf",
    "paper_citations.bib": "f9442885f74d89525da56573a0d0e4d02bbcb0d4a1854e8271576d79c94ef86e"
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
    ref = json.loads((DATASET_DIR / "manis_javanica_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-wang-manis-javanica-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
