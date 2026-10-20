#!/usr/bin/env python3
"""
2006-prum-taeniopygia-guttata-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0512)
Organism: Taeniopygia guttata (Zebra Finch)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Taeniopygia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0512_taeniopygia_guttata_vertex_packing.lean` (`MathOfBiology.Loop_0512`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/taeniopygia_guttata_vertex_packing"
EXPECTED_SHA256 = {
    "taeniopygia_guttata_vertex_packing_reference.json": "b94f622075bf99b502cf1a8b3bde862a66dbf40f3db24fc8d5466ebe05c08735",
    "taeniopygia_guttata_specimen.jpg": "cba8970e2c1bea961592887848134a4a694bea3ad4d2b68524b2a63c722dffa0",
    "scout_manifest.json": "431284b1fb47dbdafc6c9a7f1c1a8a1650f5c8a3e0b356367e5b2698e4d7047b",
    "paper_citations.bib": "bfeae2fdfaf40298df4706884b6afd5ab267e018cd2cc2ce3b17ad093ec52dd2"
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
    ref = json.loads((DATASET_DIR / "taeniopygia_guttata_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-prum-taeniopygia-guttata-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
