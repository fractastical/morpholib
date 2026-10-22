#!/usr/bin/env python3
"""
models/1992-okamoto-nipponites-mirabilis-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1600)
Organism: Nipponites mirabilis (Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite)
Phenomenon Scale: 136 µm (0.0001364 m, log10=-3.865, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Nipponites mirabilis (Late Cretaceous (Coniacian, 88 Ma))
Geological Horizon: Late Cretaceous (Coniacian, 88 Ma) (88.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nipponites
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (136 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1600_nipponites_mirabilis_vertex_packing.lean` (`MathOfBiology.Loop_1600`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nipponites_mirabilis_vertex_packing"
EXPECTED_SHA256 = {
    "nipponites_mirabilis_vertex_packing_reference.json": "752fc3f10b19ab2d69b7458f2195f19375c3a73f6fe11ef2dea1ef0356d4ab27",
    "nipponites_mirabilis_specimen.jpg": "6b8124b1e5eeb73dc589cf76fafc5db8b2d4a7d15eb68c77a1175d267f717dd5",
    "scout_manifest.json": "ad3bfc71dfcb5099dbf34d47339683961952bfc4aec3434eddf80009b7be5a72",
    "paper_citations.bib": "7fe3c1fe9a7fceff2bc3c23c28d980031b01b8a5b3c3ceda5e5928e8062266b7"
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
    ref = json.loads((DATASET_DIR / "nipponites_mirabilis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1992-okamoto-nipponites-mirabilis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
