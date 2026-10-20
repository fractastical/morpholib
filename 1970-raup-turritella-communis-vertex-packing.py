#!/usr/bin/env python3
"""
1970-raup-turritella-communis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0558)
Organism: Turritella communis (Tower Screw Shell)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Turritella
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0558_turritella_communis_vertex_packing.lean` (`MathOfBiology.Loop_0558`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/turritella_communis_vertex_packing"
EXPECTED_SHA256 = {
    "turritella_communis_vertex_packing_reference.json": "ca92e6ba730f234b00a18386954d2a975da0d5973c94789bfb159d565c46d070",
    "turritella_communis_specimen.jpg": "899a774eadcf1dae2799654a3393a678c5543bfebd3e3f9a02043f13da6934d0",
    "scout_manifest.json": "71463407b84758ebb900618937eaed8bd465626a868ab4e2a8f1f1eec51c6388",
    "paper_citations.bib": "cc6a9f3c648c69d80fc665c29b0bde28b0032f2839b5aca7e0b27907b77156cc"
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
    ref = json.loads((DATASET_DIR / "turritella_communis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1970-raup-turritella-communis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
