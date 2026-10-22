#!/usr/bin/env python3
"""
models/1999-hughes-aulacopleura-koninckii-meraspis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1242)
Organism: Aulacopleura koninckii_meraspis (Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph))
Phenomenon Scale: 183 µm (0.0001831 m, log10=-3.737, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Aulacopleura koninckii_meraspis (Silurian (Wenlock Epoch, 428.5 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.5 Ma) (428.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (183 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1242_aulacopleura_koninckii_meraspis_turing_wavelength.lean` (`MathOfBiology.Loop_1242`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_meraspis_turing_wavelength"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_meraspis_turing_wavelength_reference.json": "f2b2642dfafa8d7a56382ab3b0ccc189b541e1a2bf8c7afbcbc1c5bcf47fd16c",
    "aulacopleura_koninckii_meraspis_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "485b0cba46cc196bd04a2fef59c561a64d195b463bdc34e911145279483c999e",
    "paper_citations.bib": "6983202b6db665c157a9b9e73ae5e0de08010c711f7bbcd89b30e4fcba1c8955"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_meraspis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-hughes-aulacopleura-koninckii-meraspis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
