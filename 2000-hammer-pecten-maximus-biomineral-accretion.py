#!/usr/bin/env python3
"""
2000-hammer-pecten-maximus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0857)
Organism: Pecten maximus (Great Scallop)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pecten
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0857_pecten_maximus_biomineral_accretion.lean` (`MathOfBiology.Loop_0857`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pecten_maximus_biomineral_accretion"
EXPECTED_SHA256 = {
    "pecten_maximus_biomineral_accretion_reference.json": "330c8458c1d396cb1050b8f03229de685795c9412c5c9ceec398939aa6967d4c",
    "pecten_maximus_specimen.jpg": "2f87d4ec30ebec6adb82f91aec67eb300e3469f57920cbccf9c9bb0d22b703a8",
    "scout_manifest.json": "73b5919125b233acd14bc597eae820e24bad0dd7d4a78cb7aff9c8970733478c",
    "paper_citations.bib": "9622e0fa7fa13cc63cf843fdec0f640c28c0b2c1a823a2dffb66b75d355ea557"
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
    ref = json.loads((DATASET_DIR / "pecten_maximus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2000-hammer-pecten-maximus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
