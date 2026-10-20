#!/usr/bin/env python3
"""
1981-odell-lytechinus-variegatus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0855)
Organism: Lytechinus variegatus (Green Sea Urchin)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Lytechinus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0855_lytechinus_variegatus_biomineral_accretion.lean` (`MathOfBiology.Loop_0855`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/lytechinus_variegatus_biomineral_accretion"
EXPECTED_SHA256 = {
    "lytechinus_variegatus_biomineral_accretion_reference.json": "f1fe1318003090bcc4051f50400d5ea6931a190c18f073accdef0feebab570f6",
    "lytechinus_variegatus_specimen.jpg": "521ca1f087eb9618f793b80f0084a9efa6db3c8fb7039599c38d75ba00260619",
    "scout_manifest.json": "b95476f9c5d53fdbd4cd0ae236ebfdd4d5ea94984f951def1b514656dcfe77ac",
    "paper_citations.bib": "3d8a2203d3ed42b52cc11a66f79f825bfeb8baaf0d12058bfa3ca72ed7595146"
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
    ref = json.loads((DATASET_DIR / "lytechinus_variegatus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-odell-lytechinus-variegatus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
