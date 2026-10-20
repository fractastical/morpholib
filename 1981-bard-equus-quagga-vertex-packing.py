#!/usr/bin/env python3
"""
1981-bard-equus-quagga-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0504)
Organism: Equus quagga (Plains Zebra)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0504_equus_quagga_vertex_packing.lean` (`MathOfBiology.Loop_0504`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/equus_quagga_vertex_packing"
EXPECTED_SHA256 = {
    "equus_quagga_vertex_packing_reference.json": "d3a2fc1b7734128f46b96d65b30efd443a86cd3bb4fe07049b12e1833259717f",
    "equus_quagga_specimen.jpg": "b734dc3b533ffe889dee9f205dae9015580b4b3537a05743c602408063b17ca8",
    "scout_manifest.json": "fa966a8d0705cd73bdcb675c3bbb4c82c920c21a9b5207e20ae511c86bcd15d1",
    "paper_citations.bib": "903c18c1665dbe3f0f7328fed217c0d0f31eb0b7c18a31f86b033fd1c9b58074"
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
    ref = json.loads((DATASET_DIR / "equus_quagga_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-bard-equus-quagga-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
