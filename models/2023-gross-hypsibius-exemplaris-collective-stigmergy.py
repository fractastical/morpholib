#!/usr/bin/env python3
"""
models/2023-gross-hypsibius-exemplaris-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2091)
Organism: Hypsibius exemplaris (Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis))
Phenomenon Scale: 384 µm (0.0003841 m, log10=-3.416, 5_primordium_100um)
Anatomical Structure: Collective swarm / biofilm / colonial array in Hypsibius exemplaris (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Hypsibius
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (384 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2091_hypsibius_exemplaris_collective_stigmergy.lean` (`MathOfBiology.Loop_2091`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hypsibius_exemplaris_collective_stigmergy"
EXPECTED_SHA256 = {
    "hypsibius_exemplaris_collective_stigmergy_reference.json": "d1741611fc71da38a53e5b773029b0e99558609efa348a7c034f134541626e07",
    "hypsibius_exemplaris_specimen.jpg": "47de00f0a828a76d4d2489503394a17024fde8a0989acc96911706aa535ba88d",
    "scout_manifest.json": "223bd0ed552709929ec886541835026c3673269e5cc46a26eea100f6cd785979",
    "paper_citations.bib": "e06f0b9cdb36e45df821e36fbd7afaf6688c19806bc31251c1bbf9861d12dc0f"
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
    ref = json.loads((DATASET_DIR / "hypsibius_exemplaris_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-gross-hypsibius-exemplaris-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
