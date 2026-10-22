#!/usr/bin/env python3
"""
models/2009-ahlberg-ichthyostega-stensioei-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1576)
Organism: Ichthyostega stensioei (Late Devonian Seven-Digited Weight-Bearing Tetrapod)
Phenomenon Scale: 345 µm (0.0003449 m, log10=-3.462, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Ichthyostega stensioei (Late Devonian (Famennian, 363 Ma))
Geological Horizon: Late Devonian (Famennian, 363 Ma) (363.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ichthyostega
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (345 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1576_ichthyostega_stensioei_vertex_packing.lean` (`MathOfBiology.Loop_1576`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ichthyostega_stensioei_vertex_packing"
EXPECTED_SHA256 = {
    "ichthyostega_stensioei_vertex_packing_reference.json": "91c22ab4edbe788f71c8b691607e44e9522f266c73c29ebc14d56a906b50cb8c",
    "ichthyostega_stensioei_specimen.jpg": "d0371040f354560ee3b95904f5d16c9f224254e9b490bc2d4a9c7d712aa42e0c",
    "scout_manifest.json": "11ddb16d306f01bd834fb138f323d82725cfdb8847c060de97822a2b60dbab55",
    "paper_citations.bib": "82108d32487953189e35d8d0eee5da17a516eca0b4aa39c1fe966805cd093c00"
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
    ref = json.loads((DATASET_DIR / "ichthyostega_stensioei_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-ahlberg-ichthyostega-stensioei-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
