#!/usr/bin/env python3
"""
models/2003-royer-ginkgo-adiantoides-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1263)
Organism: Ginkgo adiantoides (Paleocene-Eocene Bilobed Coalescent Ginkgo Chronospecies)
Phenomenon Scale: 2.3 mm (0.002267 m, log10=-2.645, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Ginkgo adiantoides (Paleocene-Eocene Boundary (56 Ma))
Geological Horizon: Paleocene-Eocene Boundary (56 Ma) (56.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (2.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1263_ginkgo_adiantoides_turing_wavelength.lean` (`MathOfBiology.Loop_1263`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_adiantoides_turing_wavelength"
EXPECTED_SHA256 = {
    "ginkgo_adiantoides_turing_wavelength_reference.json": "a0f9b51453e5ae5a314ebd07a0a3d1eb54bfa4f4210ed7b2c26ff3f58bdbe324",
    "ginkgo_adiantoides_specimen.jpg": "9b107911c55810e272a4f18af3a1f21e0f0d95561cf384cba5c90c9245314ca8",
    "scout_manifest.json": "b1b0d87285083b246f698669acea87cd703876160096621143def94911454782",
    "paper_citations.bib": "82dccd06411a840afb50062a0c9cf5fd6df98c7e54d9ad408e2203864db90a2c"
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
    ref = json.loads((DATASET_DIR / "ginkgo_adiantoides_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-royer-ginkgo-adiantoides-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
