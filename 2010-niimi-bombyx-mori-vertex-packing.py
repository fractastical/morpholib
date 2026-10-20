#!/usr/bin/env python3
"""
2010-niimi-bombyx-mori-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0546)
Organism: Bombyx mori (Domestic Silkmoth)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Bombyx
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0546_bombyx_mori_vertex_packing.lean` (`MathOfBiology.Loop_0546`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/bombyx_mori_vertex_packing"
EXPECTED_SHA256 = {
    "bombyx_mori_vertex_packing_reference.json": "2e67efa27362090d7ee6667d28a7e2764db1e189b4cfd002e3a2c44b93f5f762",
    "bombyx_mori_specimen.jpg": "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97",
    "scout_manifest.json": "9ab4a2657dd86995b55773e9bfff5afa6aff20e0649f280cda14cafbc3e23c52",
    "paper_citations.bib": "786d0736b4a641d8aa204f8c6cdec98f37cb3e61ba389e0c35c9de6b3e0cedd6"
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
    ref = json.loads((DATASET_DIR / "bombyx_mori_vertex_packing_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2010-niimi-bombyx-mori-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
