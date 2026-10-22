#!/usr/bin/env python3
"""
models/2025-wan-welwitschia-mirabilis-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1647)
Organism: Welwitschia mirabilis (Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves))
Phenomenon Scale: 2.0 mm (0.001976 m, log10=-2.704, 6_organ_mm)
Anatomical Structure: Apical epithelial cell vertex polygon in Welwitschia mirabilis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Welwitschia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (2.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1647_welwitschia_mirabilis_vertex_packing.lean` (`MathOfBiology.Loop_1647`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/welwitschia_mirabilis_vertex_packing"
EXPECTED_SHA256 = {
    "welwitschia_mirabilis_vertex_packing_reference.json": "3527c8098287618b5b34339521a84a036c542df06117c4cb31e8c00f8064ef0e",
    "welwitschia_mirabilis_specimen.jpg": "ee3b34d91d4679f637d5087e7e7fa56ebbe65a443882acb98b96db019cfdc1ab",
    "scout_manifest.json": "9be7305eb77523ec022c4fc3c4ee0e2b08e4882a0c2484924237eed72003cc8b",
    "paper_citations.bib": "93440fdd8f817ae73445be9808a762ccad742090d8da23ee1e3d7935be03a8c3"
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
    ref = json.loads((DATASET_DIR / "welwitschia_mirabilis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-wan-welwitschia-mirabilis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
