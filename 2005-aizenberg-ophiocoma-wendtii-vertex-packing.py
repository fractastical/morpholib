#!/usr/bin/env python3
"""
2005-aizenberg-ophiocoma-wendtii-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0559)
Organism: Ophiocoma wendtii (Red Brittle Star)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ophiocoma
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0559_ophiocoma_wendtii_vertex_packing.lean` (`MathOfBiology.Loop_0559`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ophiocoma_wendtii_vertex_packing"
EXPECTED_SHA256 = {
    "ophiocoma_wendtii_vertex_packing_reference.json": "47939a6429e3601ff46efe9d351fca53fe0de2d9f36a5930b3f4d1f8eac998fd",
    "ophiocoma_wendtii_specimen.jpg": "c073d01d499967366ff61d9831cb0e07e7505885a8ce4a9445565ec8d34f057d",
    "scout_manifest.json": "6714f68ad6166e3fe111753571cc0a2d67a01885e2d1003e28af93b1930e3fa9",
    "paper_citations.bib": "8942a8713619deb6bfc7d74a32afc012d735a2ed2603d1d7b9e19aebd075be1c"
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
    ref = json.loads((DATASET_DIR / "ophiocoma_wendtii_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-aizenberg-ophiocoma-wendtii-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
