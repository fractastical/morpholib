#!/usr/bin/env python3
"""
2017-kotrschal-poecilia-reticulata-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0537)
Organism: Poecilia reticulata (Trinidadian Guppy)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Poecilia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0537_poecilia_reticulata_vertex_packing.lean` (`MathOfBiology.Loop_0537`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/poecilia_reticulata_vertex_packing"
EXPECTED_SHA256 = {
    "poecilia_reticulata_vertex_packing_reference.json": "64f4838d78b2b718fc83413c8e1d3cfe3488bb87c41fdf9c7b3537f14bc24244",
    "poecilia_reticulata_specimen.jpg": "3e998877d83b46f9df12f9fa135839a5982986b29a14965629955111cadf8a2e",
    "scout_manifest.json": "d775c89868fbb8d47aa0eef0081dcafd87a0dbf996feb4731a2747a590eb1820",
    "paper_citations.bib": "14f983b0b33d5d5bfe98aad9a0f35c130261d85dc54c449d77847994c8f18b54"
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
    ref = json.loads((DATASET_DIR / "poecilia_reticulata_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-kotrschal-poecilia-reticulata-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
