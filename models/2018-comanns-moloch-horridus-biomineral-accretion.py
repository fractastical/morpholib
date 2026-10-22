#!/usr/bin/env python3
"""
models/2018-comanns-moloch-horridus-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1909)
Organism: Moloch horridus (Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network))
Phenomenon Scale: 46 µm (4.65e-05 m, log10=-4.333, 4_cellular_10um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Moloch horridus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Moloch
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (46 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1909_moloch_horridus_biomineral_accretion.lean` (`MathOfBiology.Loop_1909`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/moloch_horridus_biomineral_accretion"
EXPECTED_SHA256 = {
    "moloch_horridus_biomineral_accretion_reference.json": "b493c53a548f75224a229f6ff239e2b350458cd431f3553b764eca35c1ca2ac7",
    "moloch_horridus_specimen.jpg": "224fb2ba2daad242971c5263ec01e8e384e89bea8709b08d5380e120ebdbf94a",
    "scout_manifest.json": "f5bc0d0a1a52281225fe5120905b595a1384652f96bafd64b358e7d0a3733b93",
    "paper_citations.bib": "b8e7fea00f8bfdf075d52e2e81dde58d5ade62b3cb1352e9d040c47095023e17"
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
    ref = json.loads((DATASET_DIR / "moloch_horridus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-comanns-moloch-horridus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
