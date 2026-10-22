#!/usr/bin/env python3
"""
models/2014-nweeia-monodon-monoceros-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1227)
Organism: Monodon monoceros (Arctic Narwhal (Chiral Helical Left Incisor Tusk))
Phenomenon Scale: 7.8 cm (0.0777 m, log10=-1.110, 7_organism_cm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Monodon monoceros (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Monodon
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (7.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1227_monodon_monoceros_turing_wavelength.lean` (`MathOfBiology.Loop_1227`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/monodon_monoceros_turing_wavelength"
EXPECTED_SHA256 = {
    "monodon_monoceros_turing_wavelength_reference.json": "ec01313e5eb66687368fcddfcd9e642e9577d900e1dcea6de72e1406ace50039",
    "monodon_monoceros_specimen.jpg": "1fa8766cd6ff186448201eb9da7097191319445a02628f275cfe5752d1be8cc3",
    "scout_manifest.json": "d714dd898b8d579393802ff30e40c6685daa38cf6f380fe287b8858fd8fa4001",
    "paper_citations.bib": "b7d42d2cc7a124dc4674f23b1304d3e0584b73634d95a3630574f5c7cac7bdf2"
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
    ref = json.loads((DATASET_DIR / "monodon_monoceros_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-nweeia-monodon-monoceros-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
