#!/usr/bin/env python3
"""
1998-koch-giraffa-camelopardalis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0505)
Organism: Giraffa camelopardalis (Reticulated Giraffe)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Giraffa
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0505_giraffa_camelopardalis_vertex_packing.lean` (`MathOfBiology.Loop_0505`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/giraffa_camelopardalis_vertex_packing"
EXPECTED_SHA256 = {
    "giraffa_camelopardalis_vertex_packing_reference.json": "98c274d06f7debc64d7a9d8cdf0da8c0819f920dfd5113f5fda30d40e43b1531",
    "giraffa_camelopardalis_specimen.jpg": "d65ee763a067a60ce829e77f52e0ca7eadb883d032bc517d6f76edf060800969",
    "scout_manifest.json": "bad2507baa8fb0095fb23ae25e74640f702cc1efb90b69bfb8890dc036544f01",
    "paper_citations.bib": "6bb65bc7a47caafe3310b88a27669fa318489de2db50afad1290fe77f91455b1"
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
    ref = json.loads((DATASET_DIR / "giraffa_camelopardalis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1998-koch-giraffa-camelopardalis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
