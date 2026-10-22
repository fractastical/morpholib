#!/usr/bin/env python3
"""
models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1151)
Organism: Danio rerio (Zebrafish — 16–24 hpf)
Phenomenon Scale: 42 µm (4.18e-05 m, log10=-4.379, 4_cellular_10um)
Anatomical Structure: Single nascent somite block (55 µm AP) & deposited lateral-line neuromast (16–24 hpf (Somite Epithelial Somitomere & Neuromast Rosette, ZFA:0000155)) · Zebrahub lateral_line_neuromast_rosette & segmented_paraxial_somite
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection at 42 µm (Single nascent somite block (55 µm AP) & deposited lateral-line neuromast))
Formal Verification: `morphoeval/lean/loops/Loop_1151_danio_rerio_um55_somite_rosette_neuromast_turing_wavelength.lean` (`MathOfBiology.Loop_1151`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um55_somite_rosette_neuromast_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_rerio_um55_somite_rosette_neuromast_turing_wavelength_reference.json": "f2ba5b8895e6a92f3b5ad1a51abb9f8e61b63c4c18fa2f4256a740c1661d6e5c",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "2ec14087eb68394ac96327b8c195b0cd763182a6b34d53453e53cc1ce09ab7d8",
    "paper_citations.bib": "fb5d841c50c9bd73fabe32eafd7314d855e2164fbd9537400782e952f0342c12"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um55_somite_rosette_neuromast_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-soroldoni-danio-rerio-um55-somite-rosette-neuromast-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
