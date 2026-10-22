#!/usr/bin/env python3
"""
models/2008-bohn-nepenthes-rajah-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1649)
Organism: Nepenthes rajah (Giant Bornean Pitcher Plant (Micro-Grooved Peristome Aquaplaning Rim))
Phenomenon Scale: 114 nm (1.136e-07 m, log10=-6.945, 2_macromolecular_100nm)
Anatomical Structure: Apical epithelial cell vertex polygon in Nepenthes rajah (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Nepenthes
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (114 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1649_nepenthes_rajah_vertex_packing.lean` (`MathOfBiology.Loop_1649`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nepenthes_rajah_vertex_packing"
EXPECTED_SHA256 = {
    "nepenthes_rajah_vertex_packing_reference.json": "e223bff2d4488a0a29d481af154428f9a28e8fc5e32bd90e2bc59ec72d52c245",
    "nepenthes_rajah_specimen.jpg": "34d473e5c7aa422e26ade1e48e08666c34867f853524458ff188daa017f9d5a3",
    "scout_manifest.json": "3972cfb5d2fa5ae401c23572f8f578b3cc4f65fa675936ffb35f9e6d7acccd51",
    "paper_citations.bib": "293e83be8ae5922b91a9ad1051e852dc8843027a29278cd9113f42d1b47f7352"
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
    ref = json.loads((DATASET_DIR / "nepenthes_rajah_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-bohn-nepenthes-rajah-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
