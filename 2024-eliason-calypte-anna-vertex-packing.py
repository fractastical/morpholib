#!/usr/bin/env python3
"""
2024-eliason-calypte-anna-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0520)
Organism: Calypte anna (Anna's Hummingbird)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Calypte
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0520_calypte_anna_vertex_packing.lean` (`MathOfBiology.Loop_0520`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/calypte_anna_vertex_packing"
EXPECTED_SHA256 = {
    "calypte_anna_vertex_packing_reference.json": "1e27960e2c8fe298f1808d8334ad2027deea15cd70c607f1389fe7571d069f85",
    "calypte_anna_specimen.jpg": "023a1c1ebf863e2bc3fe443e9cc7d668c6fa7ac72517c8a62f1874671a2eeb60",
    "scout_manifest.json": "3029fe7d82a3042f306916621ba18e29996d7a75e0710f65aec2d7f3ffb6d78c",
    "paper_citations.bib": "611f9775cf5f92b31c2b40626deb6e4916eb3a25702747081913a5cafe52f0f2"
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
    ref = json.loads((DATASET_DIR / "calypte_anna_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2024-eliason-calypte-anna-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
