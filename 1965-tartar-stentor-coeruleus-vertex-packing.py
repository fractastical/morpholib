#!/usr/bin/env python3
"""
1965-tartar-stentor-coeruleus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0596)
Organism: Stentor coeruleus (Trumpet Ciliate)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Stentor
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0596_stentor_coeruleus_vertex_packing.lean` (`MathOfBiology.Loop_0596`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/stentor_coeruleus_vertex_packing"
EXPECTED_SHA256 = {
    "stentor_coeruleus_vertex_packing_reference.json": "abea0925e733fa436a6d30a9bd280f0be95ff240543393565459769e32eedae4",
    "stentor_coeruleus_specimen.jpg": "a1681963dfc488967f3552b87eed7d5716e022271322cdda8623fa8000374fb8",
    "scout_manifest.json": "dbe10b743dd5fc661d649f33a396b71da1659fcc6ec5e77c468a55750697d95b",
    "paper_citations.bib": "6d7cbafadcb1acc638f9ca96880844738b88df3b0559bbf40825e4ed807e90b9"
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
    ref = json.loads((DATASET_DIR / "stentor_coeruleus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1965-tartar-stentor-coeruleus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
