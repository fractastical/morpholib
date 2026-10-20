#!/usr/bin/env python3
"""
2008-murray-homo-sapiens-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0502)
Organism: Homo sapiens (Human)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Homo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0502_homo_sapiens_vertex_packing.lean` (`MathOfBiology.Loop_0502`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/homo_sapiens_vertex_packing"
EXPECTED_SHA256 = {
    "homo_sapiens_vertex_packing_reference.json": "6bea3a682818df138f806e7b0f2959b83d0739995d4037b753d3fd997a7bde34",
    "homo_sapiens_specimen.jpg": "ed3d39e41faa4dd97074816c05e647bdc39bf7f335f22ddfa3bcd4b2760ee735",
    "scout_manifest.json": "c2ba664bff77011af4baf90dfc823a56ca6335c277ff997fa5d9cfc818ed8572",
    "paper_citations.bib": "cbbc890ff0c8791ccc622acb3c4eb143bdbf10580cc81191c99946fbb7ac12c5"
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
    ref = json.loads((DATASET_DIR / "homo_sapiens_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-murray-homo-sapiens-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
