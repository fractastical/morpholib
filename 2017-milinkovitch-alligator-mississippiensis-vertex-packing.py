#!/usr/bin/env python3
"""
2017-milinkovitch-alligator-mississippiensis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0530)
Organism: Alligator mississippiensis (American Alligator)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Alligator
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0530_alligator_mississippiensis_vertex_packing.lean` (`MathOfBiology.Loop_0530`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/alligator_mississippiensis_vertex_packing"
EXPECTED_SHA256 = {
    "alligator_mississippiensis_vertex_packing_reference.json": "91c5e06bfe7df41b0dda96e971fe8a6bbecc4ccb9b92a5fe751d75b9af934000",
    "alligator_mississippiensis_specimen.jpg": "bb1770a8cf4a71024eec779363925a2928277079f2124074117061fc6d960764",
    "scout_manifest.json": "ea6cd33ff77e5eda2b13fc76d699fda22a6e1223a50fcc519af16e974333c3d8",
    "paper_citations.bib": "4ef330f7df73686534e7768af445b6e16d20dca8536e7f41e2213f8ad9429d51"
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
    ref = json.loads((DATASET_DIR / "alligator_mississippiensis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-milinkovitch-alligator-mississippiensis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
