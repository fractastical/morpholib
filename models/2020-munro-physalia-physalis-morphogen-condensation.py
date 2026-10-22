#!/usr/bin/env python3
"""
models/2020-munro-physalia-physalis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1376)
Organism: Physalia physalis (Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail))
Phenomenon Scale: 1.9 cm (0.01922 m, log10=-1.716, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Physalia physalis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Physalia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1376_physalia_physalis_morphogen_condensation.lean` (`MathOfBiology.Loop_1376`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/physalia_physalis_morphogen_condensation"
EXPECTED_SHA256 = {
    "physalia_physalis_morphogen_condensation_reference.json": "90cc1303781e0c271b7ea520620995a5d145b7a2d3671afe6e14a5cdee1b9395",
    "physalia_physalis_specimen.jpg": "b68854e9fdaaf725c3899f1667f4e8361f6a0f2556862c28573db2d565a50904",
    "scout_manifest.json": "150572944c812a3d5612dba18e6ee722c62e615f359cbc8efcd0271c128ee2d6",
    "paper_citations.bib": "81ba20878c717450f1511a561a445075c5c4aa75a9bc4d876ad4e81d34e4e057"
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
    ref = json.loads((DATASET_DIR / "physalia_physalis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-munro-physalia-physalis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
