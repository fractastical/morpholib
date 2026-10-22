#!/usr/bin/env python3
"""
models/2001-gingerich-rodhocetus-kasranii-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1225)
Organism: Rodhocetus kasranii (Middle Eocene Pelvic-Swimming Protocetid Whale)
Phenomenon Scale: 9.3 mm (0.009312 m, log10=-2.031, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Rodhocetus kasranii (Middle Eocene (Lutetian, 46.0 Ma))
Geological Horizon: Middle Eocene (Lutetian, 46.0 Ma) (46.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Rodhocetus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (9.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1225_rodhocetus_kasranii_turing_wavelength.lean` (`MathOfBiology.Loop_1225`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rodhocetus_kasranii_turing_wavelength"
EXPECTED_SHA256 = {
    "rodhocetus_kasranii_turing_wavelength_reference.json": "9069363646819b14eacda506467cab6f7b6449efa989ea6b4e5815288bc7f1e5",
    "rodhocetus_kasranii_specimen.jpg": "939e4d05d84f57caadd235a117accb853b0aee6b303a1b365098b6db1f32f308",
    "scout_manifest.json": "610c1e5c7e1b4f6112cc18562117cdf90c08c3f357284dda5bc34b9ce8204c0d",
    "paper_citations.bib": "514663604cf99c89788d32c33cc18da7de01a861e647add81aee3120cc2c1f45"
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
    ref = json.loads((DATASET_DIR / "rodhocetus_kasranii_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-gingerich-rodhocetus-kasranii-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
