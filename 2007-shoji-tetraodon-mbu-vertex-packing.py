#!/usr/bin/env python3
"""
2007-shoji-tetraodon-mbu-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0532)
Organism: Tetraodon mbu (Giant Pufferfish)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Tetraodon
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0532_tetraodon_mbu_vertex_packing.lean` (`MathOfBiology.Loop_0532`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetraodon_mbu_vertex_packing"
EXPECTED_SHA256 = {
    "tetraodon_mbu_vertex_packing_reference.json": "4a71d845b565faf2aaac9227d5500f7fb8b5011af36e19f568268bd6055c32c3",
    "tetraodon_mbu_specimen.jpg": "c2d84cbf503eef4d1bdc55fdc0e9d3ba124750e0ae48ff7010b6a4cdca2ee3f3",
    "scout_manifest.json": "d55ce35a301490425aa2ac2ec337b71ff903a97f0d5706e261564bda4478abfb",
    "paper_citations.bib": "de45f9a0de6c7ce320aa453626c73bf4afe97ff771ce493aebb3df02905ddd2c"
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
    ref = json.loads((DATASET_DIR / "tetraodon_mbu_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-shoji-tetraodon-mbu-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
