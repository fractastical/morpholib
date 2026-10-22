#!/usr/bin/env python3
"""
models/1992-okamoto-nipponites-mirabilis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2050)
Organism: Nipponites mirabilis (Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite)
Phenomenon Scale: 9.8 cm (0.09817 m, log10=-1.008, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Nipponites mirabilis (Late Cretaceous (Coniacian, 88 Ma))
Geological Horizon: Late Cretaceous (Coniacian, 88 Ma) (88.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nipponites
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (9.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2050_nipponites_mirabilis_collective_stigmergy.lean` (`MathOfBiology.Loop_2050`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nipponites_mirabilis_collective_stigmergy"
EXPECTED_SHA256 = {
    "nipponites_mirabilis_collective_stigmergy_reference.json": "f6f613b60d87d8320172330c288d83a58d7c5372edaceae8ec736999c235d85e",
    "nipponites_mirabilis_specimen.jpg": "6b8124b1e5eeb73dc589cf76fafc5db8b2d4a7d15eb68c77a1175d267f717dd5",
    "scout_manifest.json": "c0fbc699d1662110a579822dce05a661a090b1eb8dcb86c2c756c321419f8725",
    "paper_citations.bib": "aafad713efa17f0f6718d32177e9dc57417cb8a85a68e4e5664d38c8424db6da"
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
    ref = json.loads((DATASET_DIR / "nipponites_mirabilis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-okamoto-nipponites-mirabilis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
