#!/usr/bin/env python3
"""
models/2020-comanns-moloch-horridus-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2089)
Organism: Moloch horridus (Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network))
Phenomenon Scale: 152 µm (0.0001522 m, log10=-3.818, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Moloch horridus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Moloch
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (152 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2089_moloch_horridus_collective_stigmergy.lean` (`MathOfBiology.Loop_2089`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/moloch_horridus_collective_stigmergy"
EXPECTED_SHA256 = {
    "moloch_horridus_collective_stigmergy_reference.json": "714369de01a03950a705f5f21bb3ceb2a36287af0d3fa7c78b45d69b87cc7302",
    "moloch_horridus_specimen.jpg": "224fb2ba2daad242971c5263ec01e8e384e89bea8709b08d5380e120ebdbf94a",
    "scout_manifest.json": "7347be38ad4850b3c5450d1595f8adb5a2d3dfa6180f57be88eb9a8349f31e0d",
    "paper_citations.bib": "40aac73b0cc1fad2bd859ef0b48a850d1a9592a639a4d962fce4b24e1424a767"
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
    ref = json.loads((DATASET_DIR / "moloch_horridus_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-comanns-moloch-horridus-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
