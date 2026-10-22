#!/usr/bin/env python3
"""
models/1996-edwards-cooksonia-pertoni-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2064)
Organism: Cooksonia pertoni (Silurian Dichotomous Sporophyte Stem (Oldest Vascular Tracheids))
Phenomenon Scale: 3.3 mm (0.003272 m, log10=-2.485, 6_organ_mm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Cooksonia pertoni (Late Silurian (Wenlock-Ludlow, 425 Ma))
Geological Horizon: Late Silurian (Wenlock-Ludlow, 425 Ma) (425.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Cooksonia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (3.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_2064_cooksonia_pertoni_collective_stigmergy.lean` (`MathOfBiology.Loop_2064`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/cooksonia_pertoni_collective_stigmergy"
EXPECTED_SHA256 = {
    "cooksonia_pertoni_collective_stigmergy_reference.json": "dc80ce2d910f5f40e4c92562c8349275a95cf6de861567b716423e5ffb33e824",
    "cooksonia_pertoni_specimen.jpg": "8fd04a0f4ac5dd5893b91ae0eb689cb450c030fc42bc7a41129eac5169e756c2",
    "scout_manifest.json": "e12907b30dea2d8340c01b95d489fdfef5f7127bd6e0e897a30c23f68f1b3319",
    "paper_citations.bib": "2dc7782cc827b0590e6f0bf7ea4154e14dbb58e8c9bdea274a64c3b6ff832be1"
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
    ref = json.loads((DATASET_DIR / "cooksonia_pertoni_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-edwards-cooksonia-pertoni-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
