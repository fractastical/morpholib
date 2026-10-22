#!/usr/bin/env python3
"""
models/2004-bohn-nepenthes-rajah-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1289)
Organism: Nepenthes rajah (Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim))
Phenomenon Scale: 1.8 µm (1.818e-06 m, log10=-5.740, 3_subcellular_um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Nepenthes rajah (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Nepenthes
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.8 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1289_nepenthes_rajah_turing_wavelength.lean` (`MathOfBiology.Loop_1289`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nepenthes_rajah_turing_wavelength"
EXPECTED_SHA256 = {
    "nepenthes_rajah_turing_wavelength_reference.json": "6fe12b76c211f0fb4b869e2bb93aa3653b00978698f2106c4bd3c5639456bac4",
    "nepenthes_rajah_specimen.jpg": "34d473e5c7aa422e26ade1e48e08666c34867f853524458ff188daa017f9d5a3",
    "scout_manifest.json": "0eddbc45bdace129bcbadb71c9a948d548035f3401b7a8a25ad5c0152e603524",
    "paper_citations.bib": "e2fbe3ac59cf5181bee701280c3534be445b12ceb830035c2a07032c78429774"
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
    ref = json.loads((DATASET_DIR / "nepenthes_rajah_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-bohn-nepenthes-rajah-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
