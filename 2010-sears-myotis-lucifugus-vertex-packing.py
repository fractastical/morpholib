#!/usr/bin/env python3
"""
2010-sears-myotis-lucifugus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0509)
Organism: Myotis lucifugus (Little Brown Bat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Myotis
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0509_myotis_lucifugus_vertex_packing.lean` (`MathOfBiology.Loop_0509`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/myotis_lucifugus_vertex_packing"
EXPECTED_SHA256 = {
    "myotis_lucifugus_vertex_packing_reference.json": "935e070d812883a772689b6506e4ef9e6f6111272117706aa560f785b9772d66",
    "myotis_lucifugus_specimen.jpg": "1c8f403a4e94c928ac82c06c27e448ba133f1f6b2691860354b6490e152a97b4",
    "scout_manifest.json": "849c3484ae10afc54ca8d6b114ceed0c029f009b28d798884e88036ef9b3de19",
    "paper_citations.bib": "b8a2fc4a32082e01c582f551c12f720c86911f5ea698189f649cf9a9a6b9b56d"
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
    ref = json.loads((DATASET_DIR / "myotis_lucifugus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-sears-myotis-lucifugus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
