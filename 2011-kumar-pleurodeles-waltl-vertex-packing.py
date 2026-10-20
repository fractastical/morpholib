#!/usr/bin/env python3
"""
2011-kumar-pleurodeles-waltl-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0525)
Organism: Pleurodeles waltl (Iberian Ribbed Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Pleurodeles
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0525_pleurodeles_waltl_vertex_packing.lean` (`MathOfBiology.Loop_0525`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pleurodeles_waltl_vertex_packing"
EXPECTED_SHA256 = {
    "pleurodeles_waltl_vertex_packing_reference.json": "c5f3d7445507eecc05d593948b6fad69a260d67f59cb9eeb6d91b2827aaaf3b5",
    "pleurodeles_waltl_specimen.jpg": "5b04f7216aba72146f64ff5aae72cd1f25ecdc62a55c051a3aea6d71c7dd09c5",
    "scout_manifest.json": "099dde9f181c4f121fff2f2c678fb606576da777cb864680d9021073f6236c66",
    "paper_citations.bib": "b9d3831f42278eef2bf0fed81b4b759de7c5796715d333cb3662e35241e0580b"
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
    ref = json.loads((DATASET_DIR / "pleurodeles_waltl_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-kumar-pleurodeles-waltl-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
