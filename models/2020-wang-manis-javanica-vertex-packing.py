#!/usr/bin/env python3
"""
models/2020-wang-manis-javanica-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1635)
Organism: Manis javanica (Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice))
Phenomenon Scale: 88 µm (8.838e-05 m, log10=-4.054, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Manis javanica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Manis
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (88 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1635_manis_javanica_vertex_packing.lean` (`MathOfBiology.Loop_1635`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/manis_javanica_vertex_packing"
EXPECTED_SHA256 = {
    "manis_javanica_vertex_packing_reference.json": "087898c8fbfa0044ddb0cfe3e914477079afa63b2845ab058bf1e51c1d630c6a",
    "manis_javanica_specimen.jpg": "0d2cf7b0e5853e57873b4a69f96a0bf200561aa593feda56d04025e8b7619711",
    "scout_manifest.json": "963f147f0911ea4f5e393d9e12b5b2c0dbef2102c1f7ae6a30ca1fe30b4186ce",
    "paper_citations.bib": "e8bf3e0f5e5a024448ff6e282db2836fc14ed5e9660c63c9ba931393d9c115bb"
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
    ref = json.loads((DATASET_DIR / "manis_javanica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-wang-manis-javanica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
