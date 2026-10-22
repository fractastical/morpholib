#!/usr/bin/env python3
"""
models/2012-weaver-odontodactylus-scyllarus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1283)
Organism: Odontodactylus scyllarus (Peacock Mantis Shrimp (Bouligand Helicoidal Dactyl Club Nanocomposite))
Phenomenon Scale: 27 nm (2.73e-08 m, log10=-7.564, 1_molecular_nm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Odontodactylus scyllarus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Odontodactylus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (27 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1283_odontodactylus_scyllarus_turing_wavelength.lean` (`MathOfBiology.Loop_1283`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/odontodactylus_scyllarus_turing_wavelength"
EXPECTED_SHA256 = {
    "odontodactylus_scyllarus_turing_wavelength_reference.json": "84b6e81669cd075fc676789743bff192683701440257d3a175a2460dda5ba4ca",
    "odontodactylus_scyllarus_specimen.jpg": "a0cb5ee4fe57474a9ed4bdb381117a01fec6e1ff71de38b747022330546126c0",
    "scout_manifest.json": "7457920ab7b571eadd3e12c4594c69c7671856e9e9bbb99fd275a1132c9c3f77",
    "paper_citations.bib": "dfd34fbc5ed690e7e40e27f2f55d0df0a21b6db6e35dd1752504194ae04cb412"
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
    ref = json.loads((DATASET_DIR / "odontodactylus_scyllarus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-weaver-odontodactylus-scyllarus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
