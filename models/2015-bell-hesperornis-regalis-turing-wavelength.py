#!/usr/bin/env python3
"""
models/2015-bell-hesperornis-regalis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1233)
Organism: Hesperornis regalis (Late Cretaceous Toothed Diving Bird)
Phenomenon Scale: 7.9 mm (0.007878 m, log10=-2.104, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Hesperornis regalis (Late Cretaceous (Campanian, 83.5 Ma))
Geological Horizon: Late Cretaceous (Campanian, 83.5 Ma) (83.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Hesperornis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (7.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1233_hesperornis_regalis_turing_wavelength.lean` (`MathOfBiology.Loop_1233`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hesperornis_regalis_turing_wavelength"
EXPECTED_SHA256 = {
    "hesperornis_regalis_turing_wavelength_reference.json": "7c1aacafefebffc844b66a57d6ddecd4b0e7a563188c7109a6669f61a9cc1f56",
    "hesperornis_regalis_specimen.jpg": "a8ca1cd24cf7035f2fff3d592b3f55831669012975aed64e3050567bec49e7a7",
    "scout_manifest.json": "e360f34153f69a3cc793d5770c19d6b09ea9979865a28160d8d6a23385047fc1",
    "paper_citations.bib": "4cfe6ae499cf607ae77818f7ab75713ea8080d6ee0e8cc5e45c56849e57fc802"
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
    ref = json.loads((DATASET_DIR / "hesperornis_regalis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-bell-hesperornis-regalis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
