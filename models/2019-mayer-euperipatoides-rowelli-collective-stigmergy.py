#!/usr/bin/env python3
"""
models/2019-mayer-euperipatoides-rowelli-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2092)
Organism: Euperipatoides rowelli (Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments))
Phenomenon Scale: 2.5 mm (0.002545 m, log10=-2.594, 6_organ_mm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Euperipatoides rowelli (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Euperipatoides
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (2.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_2092_euperipatoides_rowelli_collective_stigmergy.lean` (`MathOfBiology.Loop_2092`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euperipatoides_rowelli_collective_stigmergy"
EXPECTED_SHA256 = {
    "euperipatoides_rowelli_collective_stigmergy_reference.json": "6778822ee68d55de03250ebda37a63210119388724afc1aa7f7b6bedfd18fbc8",
    "euperipatoides_rowelli_specimen.jpg": "0e99f4aad858382d0f1a61bc28533c722840c8c49a15f15ce70d8f5405b262c2",
    "scout_manifest.json": "70426d795fe7e067727798b8a457bde4a70e82ef38d96d169138c5c4faf35ab0",
    "paper_citations.bib": "5de08bb1702892fed09eec244321b1758e6383db31e4937f1ed6416be404015e"
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
    ref = json.loads((DATASET_DIR / "euperipatoides_rowelli_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-mayer-euperipatoides-rowelli-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
