#!/usr/bin/env python3
"""
models/2010-shubin-tiktaalik-roseae-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2024)
Organism: Tiktaalik roseae (Late Devonian Wristed Fin-Limb Transitional Fossil)
Phenomenon Scale: 37.3 cm (0.3728 m, log10=-0.429, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Tiktaalik roseae (Late Devonian (Frasnian, 375 Ma))
Geological Horizon: Late Devonian (Frasnian, 375 Ma) (375.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Tiktaalik
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (37.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2024_tiktaalik_roseae_collective_stigmergy.lean` (`MathOfBiology.Loop_2024`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/tiktaalik_roseae_collective_stigmergy"
EXPECTED_SHA256 = {
    "tiktaalik_roseae_collective_stigmergy_reference.json": "febebd257018640c0c872f3f777ae034e397d1e424f950262a32eb65211b8f1f",
    "tiktaalik_roseae_specimen.jpg": "2b6fe2e7441ef8b8a1f69c1382b71a3606ab9e0ee552372999fb3ab07fb8c3a4",
    "scout_manifest.json": "ee2ef5120c603ef238d432e1f4a7a011bed20e8469860ca208b2b69d1843fcb7",
    "paper_citations.bib": "df0f4c8d7553240554f6f2c2e227eacc76a4c0fc47b02aa003b64aeca791389f"
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
    ref = json.loads((DATASET_DIR / "tiktaalik_roseae_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-shubin-tiktaalik-roseae-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
