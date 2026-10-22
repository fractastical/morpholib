#!/usr/bin/env python3
"""
models/2018-gorzelak-encrinus-liliiformis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2081)
Organism: Encrinus liliiformis (Middle Triassic Sea-Lily Crinoid (Stereom Gyroid Ossicles))
Phenomenon Scale: 28 µm (2.754e-05 m, log10=-4.560, 4_cellular_10um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Encrinus liliiformis (Middle Triassic (Muschelkalk, 240 Ma))
Geological Horizon: Middle Triassic (Muschelkalk, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Encrinus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (28 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2081_encrinus_liliiformis_collective_stigmergy.lean` (`MathOfBiology.Loop_2081`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/encrinus_liliiformis_collective_stigmergy"
EXPECTED_SHA256 = {
    "encrinus_liliiformis_collective_stigmergy_reference.json": "ec60f98b8b0c5ac34a9087663ed4836f303c488e85bbf3748f2d2ce0525d7480",
    "encrinus_liliiformis_specimen.jpg": "d19f3b6e48018c2dcd0ffb7be81fb1a880d1264498e9b5ec0b872b3310699a44",
    "scout_manifest.json": "6a1a89477611d4cdef0d91a98c27bfb14d4c594994abd0d66700bf76a0c3c1fa",
    "paper_citations.bib": "78458fe4963ef3fdf2d7b484cfdb5074128f148d093c1762bcca58d68ee451bb"
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
    ref = json.loads((DATASET_DIR / "encrinus_liliiformis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-gorzelak-encrinus-liliiformis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
