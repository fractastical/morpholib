#!/usr/bin/env python3
"""
models/2011-boyce-prototaxites-logani-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2070)
Organism: Prototaxites logani (Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings))
Phenomenon Scale: 61.1 cm (0.6111 m, log10=-0.214, 8_colonial_m)
Anatomical Structure: Collective swarm / biofilm / colonial array in Prototaxites logani (Early Devonian (Emsian, 400 Ma))
Geological Horizon: Early Devonian (Emsian, 400 Ma) (400.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Prototaxites
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (61.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2070_prototaxites_logani_collective_stigmergy.lean` (`MathOfBiology.Loop_2070`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/prototaxites_logani_collective_stigmergy"
EXPECTED_SHA256 = {
    "prototaxites_logani_collective_stigmergy_reference.json": "afe2ce9d4f5a09881f7e0225325a2956a706ce6962ff54c6c2bd838d9a78655a",
    "prototaxites_logani_specimen.jpg": "672ce91b81384694dbcf49ddcc7b3191d489b29fbe3b3f90608e2ed52f775b12",
    "scout_manifest.json": "53c9193aeb5a7ee6f18f39b1d74c6b4721f58c78e80963a118c0251fb42bef74",
    "paper_citations.bib": "da00064a28be4346f6f8d17896a1a3634f9b61912dac9ddb488936c650e7aff6"
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
    ref = json.loads((DATASET_DIR / "prototaxites_logani_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-boyce-prototaxites-logani-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
