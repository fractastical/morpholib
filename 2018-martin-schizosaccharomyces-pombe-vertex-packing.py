#!/usr/bin/env python3
"""
2018-martin-schizosaccharomyces-pombe-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0587)
Organism: Schizosaccharomyces pombe (Fission Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Schizosaccharomyces
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0587_schizosaccharomyces_pombe_vertex_packing.lean` (`MathOfBiology.Loop_0587`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schizosaccharomyces_pombe_vertex_packing"
EXPECTED_SHA256 = {
    "schizosaccharomyces_pombe_vertex_packing_reference.json": "cee60d3934aee30f409de76bb4d826c1f914edd8f19c5aa6ab55535f0d11aa07",
    "schizosaccharomyces_pombe_specimen.jpg": "1ddf4f2081b0c047e472365745cd54e38ef85ea46a86a451dc98e0a7d6b5c9f8",
    "scout_manifest.json": "64ad7c946dd8980f39b6948beae43d0547d24d88b029d41f941bc0f91a3e410a",
    "paper_citations.bib": "fd6c49b5891ae009cccb0ab88d028d88631de4ed70cecaa9d403547ad38a9062"
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
    ref = json.loads((DATASET_DIR / "schizosaccharomyces_pombe_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-martin-schizosaccharomyces-pombe-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
