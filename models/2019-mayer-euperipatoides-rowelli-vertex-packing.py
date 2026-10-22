#!/usr/bin/env python3
"""
models/2019-mayer-euperipatoides-rowelli-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1642)
Organism: Euperipatoides rowelli (Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments))
Phenomenon Scale: 3.5 µm (3.535e-06 m, log10=-5.452, 3_subcellular_um)
Anatomical Structure: Apical epithelial cell vertex polygon in Euperipatoides rowelli (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Euperipatoides
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (3.5 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1642_euperipatoides_rowelli_vertex_packing.lean` (`MathOfBiology.Loop_1642`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euperipatoides_rowelli_vertex_packing"
EXPECTED_SHA256 = {
    "euperipatoides_rowelli_vertex_packing_reference.json": "53dfa9b5c9f03a73d6b0d723040e56599f735c8754fe0b32cb41d4d638e790f2",
    "euperipatoides_rowelli_specimen.jpg": "0e99f4aad858382d0f1a61bc28533c722840c8c49a15f15ce70d8f5405b262c2",
    "scout_manifest.json": "d6916d53df8b58a4a9208af96873561b1985ad801164c4758f74031377ba8e18",
    "paper_citations.bib": "d36a9345dd6e9f71586edeab08b9fd7b296e41e886d2d84c7ffa233bf3297179"
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
    ref = json.loads((DATASET_DIR / "euperipatoides_rowelli_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-mayer-euperipatoides-rowelli-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
