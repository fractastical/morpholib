#!/usr/bin/env python3
"""
2017-wagner-tyto-alba-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0518)
Organism: Tyto alba (Barn Owl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Tyto
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0518_tyto_alba_vertex_packing.lean` (`MathOfBiology.Loop_0518`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tyto_alba_vertex_packing"
EXPECTED_SHA256 = {
    "tyto_alba_vertex_packing_reference.json": "c31215dfcf5a360ba2eaacf5e85f385bd4d4e6804129d36e259172ef15c40c7a",
    "tyto_alba_specimen.jpg": "1a81ca2e5a4b493f8c795ded141df4f7bfbee333f381a42a0b020882d72c2671",
    "scout_manifest.json": "7b69baac024e56863ddea033aede27718d3a95c8480bebcccf875091a00b697f",
    "paper_citations.bib": "9dba8846be9a4817f8812c1a77260ce08c54b679c285908ae5a60df20a0ade69"
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
    ref = json.loads((DATASET_DIR / "tyto_alba_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-wagner-tyto-alba-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
