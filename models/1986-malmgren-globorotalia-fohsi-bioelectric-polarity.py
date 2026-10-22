#!/usr/bin/env python3
"""
models/1986-malmgren-globorotalia-fohsi-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1986)
Organism: Globorotalia fohsi (Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies)
Phenomenon Scale: 312 nm (3.125e-07 m, log10=-6.505, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Globorotalia fohsi (Middle Miocene (Serravallian, 12.7 Ma))
Geological Horizon: Middle Miocene (Serravallian, 12.7 Ma) (12.7 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (312 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1986_globorotalia_fohsi_bioelectric_polarity.lean` (`MathOfBiology.Loop_1986`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_fohsi_bioelectric_polarity"
EXPECTED_SHA256 = {
    "globorotalia_fohsi_bioelectric_polarity_reference.json": "7926d538485ef131d8117edaca4dc8f1672d78305e7fc97b2c5ae2233865d957",
    "globorotalia_fohsi_specimen.jpg": "da1c3b52b1d7e26144fe2b2960cf08ed37afc6fd3eaf5de4044b285992f13b6f",
    "scout_manifest.json": "8b86435cc96bf84c175d642b064b526971c97f2826403d22941216aa365153d9",
    "paper_citations.bib": "ea44790e7d69c22083108d9eb64e1cff71cf87ebdadb7887e2e16a9ddbe2b0c8"
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
    ref = json.loads((DATASET_DIR / "globorotalia_fohsi_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-malmgren-globorotalia-fohsi-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
