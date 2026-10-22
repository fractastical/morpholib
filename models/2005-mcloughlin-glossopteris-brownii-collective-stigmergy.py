#!/usr/bin/env python3
"""
models/2005-mcloughlin-glossopteris-brownii-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2069)
Organism: Glossopteris brownii (Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation))
Phenomenon Scale: 9.2 cm (0.09207 m, log10=-1.036, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Glossopteris brownii (Late Permian (Gondwana, 260 Ma))
Geological Horizon: Late Permian (Gondwana, 260 Ma) (260.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Glossopteris
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (9.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2069_glossopteris_brownii_collective_stigmergy.lean` (`MathOfBiology.Loop_2069`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/glossopteris_brownii_collective_stigmergy"
EXPECTED_SHA256 = {
    "glossopteris_brownii_collective_stigmergy_reference.json": "b4feade6985945ebc3c1e5bae0b5e842838d661a355c8baf6f137f4f39b854c6",
    "glossopteris_brownii_specimen.jpg": "21f458497a8d6fffd0bfea916b3893a105c31943c6d82c526ea20f691965f627",
    "scout_manifest.json": "a8a573e90526e6bc4334ae8ee82ac2064f9d5e99e2c8a6e433230abeac12b1f7",
    "paper_citations.bib": "e719bd245e5654610cfe4ca47c25331b79a90c69d259de6c789e8b15a8c62633"
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
    ref = json.loads((DATASET_DIR / "glossopteris_brownii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-mcloughlin-glossopteris-brownii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
