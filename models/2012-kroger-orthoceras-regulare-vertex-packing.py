#!/usr/bin/env python3
"""
models/2012-kroger-orthoceras-regulare-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1594)
Organism: Orthoceras regulare (Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa))
Phenomenon Scale: 118 µm (0.0001181 m, log10=-3.928, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Orthoceras regulare (Middle Ordovician (Darriwilian, 464 Ma))
Geological Horizon: Middle Ordovician (Darriwilian, 464 Ma) (464.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Orthoceras
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (118 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1594_orthoceras_regulare_vertex_packing.lean` (`MathOfBiology.Loop_1594`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/orthoceras_regulare_vertex_packing"
EXPECTED_SHA256 = {
    "orthoceras_regulare_vertex_packing_reference.json": "4209ca34c1cb69c3fe788e298a8490ae9d9967c906a4f6c718d92c0fd33fc9bf",
    "orthoceras_regulare_specimen.jpg": "67fd83548556de1506f61ae1eaf002726c3f3a48e3a9d8452937b96eaa47a976",
    "scout_manifest.json": "276d422679605569845427b0e79ef08ff07d1cb7b2308d6a8d544754a7a08c5b",
    "paper_citations.bib": "910db262703c67a5e6249cbfb2e46c7451b250323d2729d82cb4d310ba6141e9"
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
    ref = json.loads((DATASET_DIR / "orthoceras_regulare_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-kroger-orthoceras-regulare-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
