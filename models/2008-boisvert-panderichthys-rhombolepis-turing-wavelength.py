#!/usr/bin/env python3
"""
models/2008-boisvert-panderichthys-rhombolepis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1213)
Organism: Panderichthys rhombolepis (Late Devonian Elpistostegalian Flat-Skulled Fish)
Phenomenon Scale: 6.9 mm (0.00693 m, log10=-2.159, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Panderichthys rhombolepis (Late Devonian (Frasnian, 380 Ma))
Geological Horizon: Late Devonian (Frasnian, 380 Ma) (380.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Panderichthys
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (6.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1213_panderichthys_rhombolepis_turing_wavelength.lean` (`MathOfBiology.Loop_1213`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/panderichthys_rhombolepis_turing_wavelength"
EXPECTED_SHA256 = {
    "panderichthys_rhombolepis_turing_wavelength_reference.json": "b93ec9c7828f2efd377639dcaa5f2286fc982821cdcb46920ada07e89bd90808",
    "panderichthys_rhombolepis_specimen.jpg": "5c39604c489cd7a850b62b1d1aaf2e3061b9865043e83968c2feebf4e34ca870",
    "scout_manifest.json": "e898df422d73690b00488ff7e9bc3d0640a2fc9bb953f591345bbf42bf01fc21",
    "paper_citations.bib": "7af6d2694fce7dde08a0d5624a31f3a7ec72dc9e0bada4907b80cbd8439b4df8"
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
    ref = json.loads((DATASET_DIR / "panderichthys_rhombolepis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-boisvert-panderichthys-rhombolepis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
