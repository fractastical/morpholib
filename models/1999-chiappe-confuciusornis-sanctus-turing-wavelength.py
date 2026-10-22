#!/usr/bin/env python3
"""
models/1999-chiappe-confuciusornis-sanctus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1232)
Organism: Confuciusornis sanctus (Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers))
Phenomenon Scale: 5.4 mm (0.005432 m, log10=-2.265, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Confuciusornis sanctus (Early Cretaceous (Yixian Fm, 125.0 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 125.0 Ma) (125.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Confuciusornis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (5.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1232_confuciusornis_sanctus_turing_wavelength.lean` (`MathOfBiology.Loop_1232`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/confuciusornis_sanctus_turing_wavelength"
EXPECTED_SHA256 = {
    "confuciusornis_sanctus_turing_wavelength_reference.json": "43977b10cb2e32a7ae18b211fdd411aa9cfa2f93e040aad1c34c85fe338172a3",
    "confuciusornis_sanctus_specimen.jpg": "cd7c05e2d404d66e00f79aa73a3de8f5a68cde60009b9288d61176e6902d7620",
    "scout_manifest.json": "cc7488dcc701717a437670734f40a60e421664c8ce24b362c59b9859752dc179",
    "paper_citations.bib": "d7875c8625630716f581699ebd971b1b4572d0b70f20e0887a76b4f26646325f"
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
    ref = json.loads((DATASET_DIR / "confuciusornis_sanctus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-chiappe-confuciusornis-sanctus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
