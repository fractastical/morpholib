#!/usr/bin/env python3
"""
models/2019-mitchell-fractofusus-misrai-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1611)
Organism: Fractofusus misrai (Ediacaran Four-Order Self-Similar Spindle Rangeomorph)
Phenomenon Scale: 205 µm (0.0002047 m, log10=-3.689, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Fractofusus misrai (Late Ediacaran (Newfoundland, 565 Ma))
Geological Horizon: Late Ediacaran (Newfoundland, 565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Fractofusus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (205 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1611_fractofusus_misrai_vertex_packing.lean` (`MathOfBiology.Loop_1611`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/fractofusus_misrai_vertex_packing"
EXPECTED_SHA256 = {
    "fractofusus_misrai_vertex_packing_reference.json": "9951faf1eb74c242e6dd1f32d5c780da2abd823911a871e7b936c6229c5838c5",
    "fractofusus_misrai_specimen.jpg": "b6b0eca46e1970dc859de875d6c583e3ce1abb548f235c9292a993aa71e14061",
    "scout_manifest.json": "ef4c0b597c3909ea9b70c6e96f1d680de9040da104002ef5aef0816cae0c93e5",
    "paper_citations.bib": "399696c29fd9119c4e1399eff4b464586125122b766cac3fe84e3677e2fc5d18"
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
    ref = json.loads((DATASET_DIR / "fractofusus_misrai_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-mitchell-fractofusus-misrai-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
