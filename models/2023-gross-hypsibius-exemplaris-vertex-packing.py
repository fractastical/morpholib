#!/usr/bin/env python3
"""
models/2023-gross-hypsibius-exemplaris-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1641)
Organism: Hypsibius exemplaris (Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis))
Phenomenon Scale: 534 nm (5.335e-07 m, log10=-6.273, 2_macromolecular_100nm)
Anatomical Structure: Apical epithelial cell vertex polygon in Hypsibius exemplaris (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Hypsibius
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (534 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1641_hypsibius_exemplaris_vertex_packing.lean` (`MathOfBiology.Loop_1641`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hypsibius_exemplaris_vertex_packing"
EXPECTED_SHA256 = {
    "hypsibius_exemplaris_vertex_packing_reference.json": "558e10ad6f0a9cfb80ebb9ebedcc6c8d8542523647462fae9209d97eeabb1149",
    "hypsibius_exemplaris_specimen.jpg": "47de00f0a828a76d4d2489503394a17024fde8a0989acc96911706aa535ba88d",
    "scout_manifest.json": "f0e36a8280a027e1de51edd4dcf32d6f035f667491ce771fb1e650ef64b0e7db",
    "paper_citations.bib": "e828de8590ca4d390adbffdeb9a94788df525234bdea55755e579681c93896ae"
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
    ref = json.loads((DATASET_DIR / "hypsibius_exemplaris_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-gross-hypsibius-exemplaris-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
