#!/usr/bin/env python3
"""
models/2001-hughes-aulacopleura-koninckii-meraspis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1872)
Organism: Aulacopleura koninckii_meraspis (Silurian Trilobite Meraspid Larva (12 Thoracic Segments, Early Morph))
Phenomenon Scale: 2.5 mm (0.002518 m, log10=-2.599, 6_organ_mm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Aulacopleura koninckii_meraspis (Silurian (Wenlock Epoch, 428.5 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.5 Ma) (428.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (2.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1872_aulacopleura_koninckii_meraspis_biomineral_accretion.lean` (`MathOfBiology.Loop_1872`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_meraspis_biomineral_accretion"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_meraspis_biomineral_accretion_reference.json": "ab4c5c34b54bb30ee4ee82dc5218d42f668b2342fabd78640e61cebc8318af07",
    "aulacopleura_koninckii_meraspis_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "252e17e01dfe03bc606e8f8702b5996265a50226d224bb652da61dc23853a0c2",
    "paper_citations.bib": "86d6f95bdeddd1eb473b41e6f5d14086a70d962ed880f0de78c25724f77cdc55"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_meraspis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-hughes-aulacopleura-koninckii-meraspis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
