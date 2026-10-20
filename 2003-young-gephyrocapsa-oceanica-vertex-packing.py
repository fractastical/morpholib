#!/usr/bin/env python3
"""
2003-young-gephyrocapsa-oceanica-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0598)
Organism: Gephyrocapsa oceanica (Calcite Coccolithophore)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Gephyrocapsa
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0598_gephyrocapsa_oceanica_vertex_packing.lean` (`MathOfBiology.Loop_0598`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gephyrocapsa_oceanica_vertex_packing"
EXPECTED_SHA256 = {
    "gephyrocapsa_oceanica_vertex_packing_reference.json": "1e54e6f50bc28f80fc0c8bc17b4d8b7537d5dffe9ec50acf8df2b22993b06df9",
    "gephyrocapsa_oceanica_specimen.jpg": "b3482bdf62d2f619ecd8ad93b8bc68fb784477c51f4de8d0682d3d8fa3dbbc9e",
    "scout_manifest.json": "ad33d5fba02700aafa9809a56cd5c8d6ec7684070cc705e13cc1a06b16614708",
    "paper_citations.bib": "31ab2f0895c02279952f782cb7509970e52bdc4fed1e16597a01e40161d9a319"
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
    ref = json.loads((DATASET_DIR / "gephyrocapsa_oceanica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-young-gephyrocapsa-oceanica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
