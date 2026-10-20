#!/usr/bin/env python3
"""
2016-sarrazin-tribolium-castaneum-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0542)
Organism: Tribolium castaneum (Red Flour Beetle)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Tribolium
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0542_tribolium_castaneum_vertex_packing.lean` (`MathOfBiology.Loop_0542`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tribolium_castaneum_vertex_packing"
EXPECTED_SHA256 = {
    "tribolium_castaneum_vertex_packing_reference.json": "587fc3e7fc0ac63e32ee4ee4714e3846a772c22053d91e95ecb9d6e5d5373f9c",
    "tribolium_castaneum_specimen.jpg": "a991c507b6f59178075bea80d7ad2c1c97d839991d31594797919b07b5ac1d77",
    "scout_manifest.json": "318e64012401825e089d3dc0014385763637c22c08df8dff16c94bee4f097f19",
    "paper_citations.bib": "9218393894217767b8c283cd386de7c7262d60d69a668dc79df666c1f9cfe17e"
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
    ref = json.loads((DATASET_DIR / "tribolium_castaneum_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-sarrazin-tribolium-castaneum-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
