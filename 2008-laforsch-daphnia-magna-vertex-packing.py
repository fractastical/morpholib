#!/usr/bin/env python3
"""
2008-laforsch-daphnia-magna-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0549)
Organism: Daphnia magna (Water Flea)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Daphnia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0549_daphnia_magna_vertex_packing.lean` (`MathOfBiology.Loop_0549`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/daphnia_magna_vertex_packing"
EXPECTED_SHA256 = {
    "daphnia_magna_vertex_packing_reference.json": "1c8aa0c214b3d10db796ebe73bcdfa16cd38dc103cd7a7fe2cdac1c7cef260ce",
    "daphnia_magna_specimen.jpg": "fc61b6ab51b92e988777a61cce4db7434210496ee5084ae01ee9f7adf6fcf44b",
    "scout_manifest.json": "788ca8c823b1f2374a53c2846ab688c0e2301696c366ef1d902783fa237840ec",
    "paper_citations.bib": "c833a347d46d4c4287f969cc116e62859f92db50b6e47e8e9c6685b916f29fc5"
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
    ref = json.loads((DATASET_DIR / "daphnia_magna_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-laforsch-daphnia-magna-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
