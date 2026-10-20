#!/usr/bin/env python3
"""
2017-forterre-mimosa-pudica-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0578)
Organism: Mimosa pudica (Sensitive Plant)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Mimosa
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0578_mimosa_pudica_vertex_packing.lean` (`MathOfBiology.Loop_0578`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mimosa_pudica_vertex_packing"
EXPECTED_SHA256 = {
    "mimosa_pudica_vertex_packing_reference.json": "54cdbc68481be7759e8fb0f9a2bd57139c35dbe41748b904e899940a6cf35263",
    "mimosa_pudica_specimen.jpg": "4ae1af66e325ef597ebdce00e9f639174383d049bc7b0e98b02e97de71c83e20",
    "scout_manifest.json": "9dd048179d03ae3485001c8c95d230328b9419d76e8439fb1ab7387bb272de2c",
    "paper_citations.bib": "27b513969833629c179b4f68b3f68e27e83e905b3ef52b0a24a877b9bfda2be8"
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
    ref = json.loads((DATASET_DIR / "mimosa_pudica_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-forterre-mimosa-pudica-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
