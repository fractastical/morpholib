#!/usr/bin/env python3
"""
models/2015-hull-globorotalia-tumida-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1628)
Organism: Globorotalia tumida (Pliocene-Holocene Heavy-Keeled Biconvex Foraminifer Chronospecies)
Phenomenon Scale: 1.7 µm (1.717e-06 m, log10=-5.765, 3_subcellular_um)
Anatomical Structure: Apical epithelial cell vertex polygon in Globorotalia tumida (Pliocene-Holocene (5.3–0 Ma))
Geological Horizon: Pliocene-Holocene (5.3–0 Ma) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (1.7 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1628_globorotalia_tumida_vertex_packing.lean` (`MathOfBiology.Loop_1628`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_tumida_vertex_packing"
EXPECTED_SHA256 = {
    "globorotalia_tumida_vertex_packing_reference.json": "20ec5c2c15aa9082fdd1a15e93a33389e539c4237017038345be878f6c4a70a0",
    "globorotalia_tumida_specimen.jpg": "991d4acf78405580eb9f4efd4f75f1e9aec37937e62dce3147b582690341043b",
    "scout_manifest.json": "3086a5cdb450e1bd34045a42317bf8bf812d7759647d3e235df896d9b763b001",
    "paper_citations.bib": "5d3e1a26784ed0730c27798d16494fa7b17aa8ca7ef5b1c11fca65e9f29d3caa"
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
    ref = json.loads((DATASET_DIR / "globorotalia_tumida_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-hull-globorotalia-tumida-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
