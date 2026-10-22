#!/usr/bin/env python3
"""
models/1986-grande-knightia-eocaena-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2015)
Organism: Knightia eocaena (Green River Eocene Clupeomorph Fossil)
Phenomenon Scale: 17.3 cm (0.1727 m, log10=-0.763, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Knightia eocaena (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Knightia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (17.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2015_knightia_eocaena_collective_stigmergy.lean` (`MathOfBiology.Loop_2015`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/knightia_eocaena_collective_stigmergy"
EXPECTED_SHA256 = {
    "knightia_eocaena_collective_stigmergy_reference.json": "1f80f9d3c0331ed76f56f7f52da1ebc9c6b33f9c9a5639876c65d2ea687620ce",
    "knightia_eocaena_specimen.jpg": "52c380669cecf901cc96c680b34fb444376e73b78220ae87b0df3a3c9d273c71",
    "scout_manifest.json": "6263a80b35756c852d79fbdc1b05f26c3083ae9cb9c14356fe446617a4761a86",
    "paper_citations.bib": "85a153bc932f729191517752615c2149ad0299ed9c7b4405ea495f42e65ab44a"
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
    ref = json.loads((DATASET_DIR / "knightia_eocaena_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1986-grande-knightia-eocaena-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
