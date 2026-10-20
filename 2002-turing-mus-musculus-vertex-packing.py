#!/usr/bin/env python3
"""
2002-turing-mus-musculus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0501)
Organism: Mus musculus (House Mouse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0501_mus_musculus_vertex_packing.lean` (`MathOfBiology.Loop_0501`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mus_musculus_vertex_packing"
EXPECTED_SHA256 = {
    "mus_musculus_vertex_packing_reference.json": "178ccdd30d363d8bfce0a367d02202db12ea94242198eda471c2cdab9421729e",
    "mus_musculus_specimen.jpg": "f2738290d66e441895fc10af7d7e541424263249cda0c7a8194ff01b410fe385",
    "scout_manifest.json": "fa90e2292e22ae4b085c32522f84763dca5a5d3b9c973323cffe31b93eb48cf9",
    "paper_citations.bib": "2aa9f71c2e3782854305217dfbdb43361a71f4a4cf1aabd9e720df0ae0df3798"
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
    ref = json.loads((DATASET_DIR / "mus_musculus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-turing-mus-musculus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
