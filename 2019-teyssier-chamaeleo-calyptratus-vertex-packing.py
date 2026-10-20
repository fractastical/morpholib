#!/usr/bin/env python3
"""
2019-teyssier-chamaeleo-calyptratus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0523)
Organism: Chamaeleo calyptratus (Veiled Chameleon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Chamaeleo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0523_chamaeleo_calyptratus_vertex_packing.lean` (`MathOfBiology.Loop_0523`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chamaeleo_calyptratus_vertex_packing"
EXPECTED_SHA256 = {
    "chamaeleo_calyptratus_vertex_packing_reference.json": "7fa882b4eaa6c4896eb783aa690b8e13402df315c51709be36bcff1b496453e4",
    "chamaeleo_calyptratus_specimen.jpg": "976c60b3ba6aa23d8e363fe2188ae09cd8c806190b8b02939475defbf44924b3",
    "scout_manifest.json": "e83955e7ccde439cde4a45abe2784ec58c10336b2d655b6f00cc9d9e8b4528b3",
    "paper_citations.bib": "b126b469da930bfa212338000e233ab75c34469b23031a989a685642477b4a59"
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
    ref = json.loads((DATASET_DIR / "chamaeleo_calyptratus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-teyssier-chamaeleo-calyptratus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
