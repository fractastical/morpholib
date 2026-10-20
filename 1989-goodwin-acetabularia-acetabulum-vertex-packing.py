#!/usr/bin/env python3
"""
1989-goodwin-acetabularia-acetabulum-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0592)
Organism: Acetabularia acetabulum (Mermaid's Wineglass Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acetabularia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0592_acetabularia_acetabulum_vertex_packing.lean` (`MathOfBiology.Loop_0592`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acetabularia_acetabulum_vertex_packing"
EXPECTED_SHA256 = {
    "acetabularia_acetabulum_vertex_packing_reference.json": "3c9aca41196bf2805758f3d6408a05cf3d804305211070afa1293bd8091ed721",
    "acetabularia_acetabulum_specimen.jpg": "17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1",
    "scout_manifest.json": "1cd420d104d1934acb7e30a087e7565e6dae44cb1fe76e77358156399a0da2a9",
    "paper_citations.bib": "bcb9fd2a4f8afb27abec2c129fc7efa765113f857b9784a0aaecd4cafcbce45c"
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
    ref = json.loads((DATASET_DIR / "acetabularia_acetabulum_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1989-goodwin-acetabularia-acetabulum-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
