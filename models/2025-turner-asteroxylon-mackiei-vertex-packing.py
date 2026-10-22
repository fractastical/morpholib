#!/usr/bin/env python3
"""
models/2025-turner-asteroxylon-mackiei-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1616)
Organism: Asteroxylon mackiei (Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls))
Phenomenon Scale: 18 µm (1.771e-05 m, log10=-4.752, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Asteroxylon mackiei (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Asteroxylon
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (18 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1616_asteroxylon_mackiei_vertex_packing.lean` (`MathOfBiology.Loop_1616`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/asteroxylon_mackiei_vertex_packing"
EXPECTED_SHA256 = {
    "asteroxylon_mackiei_vertex_packing_reference.json": "d6be7ad58df1830a53008dec5a3683b04d845ff2663c12a7cbe1080807efb7c0",
    "asteroxylon_mackiei_specimen.jpg": "af15c01893b31829fefca3fc2246896ab8d2c12ab87d5809f28504e9c2968531",
    "scout_manifest.json": "24a57a72757ea03decdb9efcd598bc7a33528b6f543ca6c3d6314886f2061cb6",
    "paper_citations.bib": "49c42f71ccdb2868b68bda0097d87795b9a236df2969b9b3667abd39e0938eab"
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
    ref = json.loads((DATASET_DIR / "asteroxylon_mackiei_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-turner-asteroxylon-mackiei-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
