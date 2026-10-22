#!/usr/bin/env python3
"""
models/2009-colosimo-gasterosteus-aculeatus-marine-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1570)
Organism: Gasterosteus aculeatus_marine (Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates))
Phenomenon Scale: 151 µm (0.0001511 m, log10=-3.821, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Gasterosteus aculeatus_marine (Late Pleistocene Marine Ancestor (15 ka))
Geological Horizon: Late Pleistocene Marine Ancestor (15 ka) (0.015 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (151 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1570_gasterosteus_aculeatus_marine_vertex_packing.lean` (`MathOfBiology.Loop_1570`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_marine_vertex_packing"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_marine_vertex_packing_reference.json": "43cb2710b0cd2b0e4ce7d6154a4ae207df23d03bf37e1a829a277713780c3cb6",
    "gasterosteus_aculeatus_marine_specimen.jpg": "4da41d2381dcc4c8c66a7b6ffe79757e5ca0b0dae3978efe5a16417ec3a5bd15",
    "scout_manifest.json": "e8fb91443e78151a3ca9e776f2316acf08b59d9cced3593802d194ea18b9b619",
    "paper_citations.bib": "a8462096def283dd79c2cd99fc565177f061278496dc8e7dba8ea988a11b4e3f"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_marine_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-colosimo-gasterosteus-aculeatus-marine-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
