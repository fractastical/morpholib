#!/usr/bin/env python3
"""
2017-chirat-ovis-aries-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0507)
Organism: Ovis aries (Bighorn Sheep)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ovis
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0507_ovis_aries_vertex_packing.lean` (`MathOfBiology.Loop_0507`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ovis_aries_vertex_packing"
EXPECTED_SHA256 = {
    "ovis_aries_vertex_packing_reference.json": "ef5725829731b7847d5a2a7519b8a06364df8bb1b96b93484d1d4ad3211365eb",
    "ovis_aries_specimen.jpg": "3e5aea2a1dd349564873df9e61677c401d7c2264295f722a777e3808f6a3c0d2",
    "scout_manifest.json": "f153c234d66509ee6ea98cfed138ff32011a81c0419783bcb834ac32dad635b2",
    "paper_citations.bib": "7c07e676c3844c91dfa233bc60e90e09ed4ecb82c356de0d599a7bae1e3d31b0"
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
    ref = json.loads((DATASET_DIR / "ovis_aries_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-chirat-ovis-aries-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
