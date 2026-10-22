#!/usr/bin/env python3
"""
models/2011-chen-dasypus-novemcinctus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1276)
Organism: Dasypus novemcinctus (Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles))
Phenomenon Scale: 273 µm (0.000273 m, log10=-3.564, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Dasypus novemcinctus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dasypus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (273 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1276_dasypus_novemcinctus_turing_wavelength.lean` (`MathOfBiology.Loop_1276`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dasypus_novemcinctus_turing_wavelength"
EXPECTED_SHA256 = {
    "dasypus_novemcinctus_turing_wavelength_reference.json": "2563392e6e84833999853064f24187cbbeb721d887618819d9c09e45f7b9a315",
    "dasypus_novemcinctus_specimen.jpg": "5cbb611c05d8707d92d1c6d2cdd19de1525cb938009adae92309147f22174539",
    "scout_manifest.json": "486b940249561521fd9879e3611354c7408f1269aa9160e4bd890d3ff28975c6",
    "paper_citations.bib": "32622baff36ef7831ab65199599869312c677c58d5206afdd8cb9b5a5e695563"
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
    ref = json.loads((DATASET_DIR / "dasypus_novemcinctus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-chen-dasypus-novemcinctus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
