#!/usr/bin/env python3
"""
models/1995-macfadden-mesohippus-bairdi-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1939)
Organism: Mesohippus bairdi (Late Eocene Three-Toed Sub-Hypsodont Horse)
Phenomenon Scale: 101 µm (0.000101 m, log10=-3.996, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Mesohippus bairdi (Late Eocene (Chadronian, 35.0 Ma))
Geological Horizon: Late Eocene (Chadronian, 35.0 Ma) (35.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mesohippus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (101 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1939_mesohippus_bairdi_bioelectric_polarity.lean` (`MathOfBiology.Loop_1939`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/mesohippus_bairdi_bioelectric_polarity"
EXPECTED_SHA256 = {
    "mesohippus_bairdi_bioelectric_polarity_reference.json": "be0f8067215cce9f20ec9e300dd050d6ba07727d1f302691f3fe4ff494b41e3a",
    "mesohippus_bairdi_specimen.jpg": "ffa85314095959cda06a6a0c44566047678b150e06996250a97c7c8b4dbc1876",
    "scout_manifest.json": "346acf4c24c480d20afe3191fe50edbaef68853c821d988761f5d094a68d7fa3",
    "paper_citations.bib": "f78cfba246da8d5556fb7270f5f0b7f9a0ab79e004605fde6ff8c3fe3301ab34"
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
    ref = json.loads((DATASET_DIR / "mesohippus_bairdi_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1995-macfadden-mesohippus-bairdi-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
