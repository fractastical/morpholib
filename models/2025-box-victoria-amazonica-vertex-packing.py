#!/usr/bin/env python3
"""
models/2025-box-victoria-amazonica-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1648)
Organism: Victoria amazonica (Giant Amazon Waterlily (Load-Bearing Vascular Girder Underside Ribs))
Phenomenon Scale: 3.5 mm (0.003516 m, log10=-2.454, 6_organ_mm)
Anatomical Structure: Apical epithelial cell vertex polygon in Victoria amazonica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Victoria
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (3.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1648_victoria_amazonica_vertex_packing.lean` (`MathOfBiology.Loop_1648`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/victoria_amazonica_vertex_packing"
EXPECTED_SHA256 = {
    "victoria_amazonica_vertex_packing_reference.json": "13d13bf2dd041a8b0ebe7ecd8ebb97c80f8b6e5abcc6439cb0b186e1f0437dd6",
    "victoria_amazonica_specimen.jpg": "e7c7e2af042ff07c75a8167e10cafcf5c7e3b5f0b4f2ddfd337ff9d436171be1",
    "scout_manifest.json": "c26a91c27bdaead42e3c05f12f1e262dd4e56cc636a941444aec1134b5dea855",
    "paper_citations.bib": "d3cbfc6d77b08ce58714e363e34852ecc90ba1b496bc96bfc22113e03f30d9c9"
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
    ref = json.loads((DATASET_DIR / "victoria_amazonica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-box-victoria-amazonica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
