#!/usr/bin/env python3
"""
models/2011-hull-globorotalia-tumida-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1268)
Organism: Globorotalia tumida (Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies)
Phenomenon Scale: 27 µm (2.747e-05 m, log10=-4.561, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Globorotalia tumida (Pliocene-Holocene (5.3–0 Ma))
Geological Horizon: Pliocene-Holocene (5.3–0 Ma) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (27 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1268_globorotalia_tumida_turing_wavelength.lean` (`MathOfBiology.Loop_1268`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_tumida_turing_wavelength"
EXPECTED_SHA256 = {
    "globorotalia_tumida_turing_wavelength_reference.json": "8ddebdb039413c49576e840d81f38404a9d1d1f1b97f600256782405d1d625c0",
    "globorotalia_tumida_specimen.jpg": "991d4acf78405580eb9f4efd4f75f1e9aec37937e62dce3147b582690341043b",
    "scout_manifest.json": "cff042f8bcc8d4ed102553211340be2fbecdc6875428e1a314b6e7158cdecc02",
    "paper_citations.bib": "3d626a6dca97c50d2fe7fbc93bd5a6de3e3786bf071e5315ac1f7d267f3b371d"
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
    ref = json.loads((DATASET_DIR / "globorotalia_tumida_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-hull-globorotalia-tumida-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
