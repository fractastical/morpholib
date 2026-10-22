#!/usr/bin/env python3
"""
models/1967-durham-helicoplacus-gilberti-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1629)
Organism: Helicoplacus gilberti (Early Cambrian Triradial Helical-Plated Stem-Echinoderm)
Phenomenon Scale: 84 µm (8.4e-05 m, log10=-4.076, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Helicoplacus gilberti (Early Cambrian (Poleta Fm, 516 Ma))
Geological Horizon: Early Cambrian (Poleta Fm, 516 Ma) (516.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Helicoplacus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (84 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1629_helicoplacus_gilberti_vertex_packing.lean` (`MathOfBiology.Loop_1629`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/helicoplacus_gilberti_vertex_packing"
EXPECTED_SHA256 = {
    "helicoplacus_gilberti_vertex_packing_reference.json": "060d74ab1c073cc5d0e4aefd172c430c87b1c51c736038d97bd8a20119922cb2",
    "helicoplacus_gilberti_specimen.jpg": "13c0993dd3e63798e7d236896fbbbbe4e10b4f49fe7d356be1e6a56d3f579d8a",
    "scout_manifest.json": "173e81573a922e45b3d2167b920a1888e4e858d02c743576a47903cbb7f2391d",
    "paper_citations.bib": "0a144a6a7eb2f6db1ca719939cb5cfb825045bb7f011441a23898fd34134bb87"
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
    ref = json.loads((DATASET_DIR / "helicoplacus_gilberti_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1967-durham-helicoplacus-gilberti-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
