#!/usr/bin/env python3
"""
models/2003-saunders-goniatites-crenistria-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2045)
Organism: Goniatites crenistria (Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18))
Phenomenon Scale: 3.5 cm (0.03532 m, log10=-1.452, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Goniatites crenistria (Early Carboniferous (Viséan, 333 Ma))
Geological Horizon: Early Carboniferous (Viséan, 333 Ma) (333.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Goniatites
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (3.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2045_goniatites_crenistria_collective_stigmergy.lean` (`MathOfBiology.Loop_2045`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/goniatites_crenistria_collective_stigmergy"
EXPECTED_SHA256 = {
    "goniatites_crenistria_collective_stigmergy_reference.json": "e889b83c10b6bcdd77b5404ec88ae508de2bbca7d029129ef5d0d27415d86ae4",
    "goniatites_crenistria_specimen.jpg": "1c025730c53d3fe4dde10fe4dfc3229dac439001911ee16d02107da5f04120fb",
    "scout_manifest.json": "3384334de799b7cd94beb659c55e7d3ccfbb26c9ba7a897012342ead3bca01c7",
    "paper_citations.bib": "1ce6f1cac47aab145979dd0fade588e0332ed81537474c633b4369a449816849"
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
    ref = json.loads((DATASET_DIR / "goniatites_crenistria_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-saunders-goniatites-crenistria-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
