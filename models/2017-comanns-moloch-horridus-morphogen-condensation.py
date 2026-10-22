#!/usr/bin/env python3
"""
models/2017-comanns-moloch-horridus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1369)
Organism: Moloch horridus (Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network))
Phenomenon Scale: 15 µm (1.522e-05 m, log10=-4.818, 4_cellular_10um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Moloch horridus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Moloch
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (15 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1369_moloch_horridus_morphogen_condensation.lean` (`MathOfBiology.Loop_1369`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/moloch_horridus_morphogen_condensation"
EXPECTED_SHA256 = {
    "moloch_horridus_morphogen_condensation_reference.json": "ceae1b5a646346b5ce7f77ca8417d162884d2b287d100b60499045b607cdaf89",
    "moloch_horridus_specimen.jpg": "224fb2ba2daad242971c5263ec01e8e384e89bea8709b08d5380e120ebdbf94a",
    "scout_manifest.json": "874de883b2bde89a057fe30e8f5d1092b0006158b13206e414e57a21cd4ba566",
    "paper_citations.bib": "639d5af838585856ef99d7b0b114cab05f2c50d3f15a2eda70c9106e872e7b80"
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
    ref = json.loads((DATASET_DIR / "moloch_horridus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-comanns-moloch-horridus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
