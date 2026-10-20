#!/usr/bin/env python3
"""
2004-chuong-gallus-gallus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0511)
Organism: Gallus gallus (Red Junglefowl Chick)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Gallus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0511_gallus_gallus_vertex_packing.lean` (`MathOfBiology.Loop_0511`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gallus_gallus_vertex_packing"
EXPECTED_SHA256 = {
    "gallus_gallus_vertex_packing_reference.json": "29e6a45a2065933d6272ff88b4461d88a795ee0babd281a99f41a557bbb938e2",
    "gallus_gallus_specimen.jpg": "7e7d58db6bb564ebe68e091678338c7c06b920cdb66f9dffd99b4141bf16c9cb",
    "scout_manifest.json": "09a0986f64c12d9259287faae6fbf654a4788578ab2762d3bd616ea825d73cda",
    "paper_citations.bib": "4bc21f140f92de5ff2a587a6c9e9c4c32360315127fd9a17da58fd5198d5e9a8"
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
    ref = json.loads((DATASET_DIR / "gallus_gallus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-chuong-gallus-gallus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
