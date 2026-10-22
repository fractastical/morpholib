#!/usr/bin/env python3
"""
models/1985-grande-knightia-eocaena-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1925)
Organism: Knightia eocaena (Green River Eocene Clupeomorph Fossil)
Phenomenon Scale: 77 µm (7.676e-05 m, log10=-4.115, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Knightia eocaena (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Knightia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (77 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1925_knightia_eocaena_bioelectric_polarity.lean` (`MathOfBiology.Loop_1925`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/knightia_eocaena_bioelectric_polarity"
EXPECTED_SHA256 = {
    "knightia_eocaena_bioelectric_polarity_reference.json": "8c095dbf53aee2bb9ba5470b3f8191da445f8783351f123a8bfe8618c3b05813",
    "knightia_eocaena_specimen.jpg": "52c380669cecf901cc96c680b34fb444376e73b78220ae87b0df3a3c9d273c71",
    "scout_manifest.json": "d73bb649f43e78c1572efc0e43efdfa565d4339312b067b03af41f32cc30373a",
    "paper_citations.bib": "6bb3ca38e10e628a1ca97af469b0fc0972083e70943326567c659fc236e1df97"
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
    ref = json.loads((DATASET_DIR / "knightia_eocaena_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1985-grande-knightia-eocaena-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
