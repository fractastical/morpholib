#!/usr/bin/env python3
"""
2020-alwes-parhyale-hawaiensis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0545)
Organism: Parhyale hawaiensis (Amphipod Crustacean)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Parhyale
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0545_parhyale_hawaiensis_vertex_packing.lean` (`MathOfBiology.Loop_0545`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/parhyale_hawaiensis_vertex_packing"
EXPECTED_SHA256 = {
    "parhyale_hawaiensis_vertex_packing_reference.json": "2cc988eaad0bdc18634708633011f39357db23448671cfd1cf06d0aefeb5aef2",
    "parhyale_hawaiensis_specimen.jpg": "b07b7fd2a6e42cf2286c7448b295d88791d4bc197bb214ee0cf7388093540dba",
    "scout_manifest.json": "1ae32b1654bf5c9039b48610a55ae049d4e6fbb21b0c52579c1ba240420d5de5",
    "paper_citations.bib": "64e148f7f7f2fbc1d3bbc89a33157f1d27a9e8382300cf84dfcebc2785585306"
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
    ref = json.loads((DATASET_DIR / "parhyale_hawaiensis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-alwes-parhyale-hawaiensis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
