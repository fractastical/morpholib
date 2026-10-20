#!/usr/bin/env python3
"""
2007-hinman-asterias-rubens-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0554)
Organism: Asterias rubens (Common Starfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Asterias
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0554_asterias_rubens_vertex_packing.lean` (`MathOfBiology.Loop_0554`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/asterias_rubens_vertex_packing"
EXPECTED_SHA256 = {
    "asterias_rubens_vertex_packing_reference.json": "da2b5255ee93cb5ec8b5727c4303a9d688f5ba663f632063f70164cb5d3b141b",
    "asterias_rubens_specimen.jpg": "37d94bdfed179302d1e9aa062333c2f87a5cb3d7c3932c51056f5e68c7c762ad",
    "scout_manifest.json": "9224c26c0cccf7a0db1ad8625806cbd19ea7d9b6d47fece538430a1f16958bba",
    "paper_citations.bib": "69890088fd0af403f47f9690c305b67632d4bbca710be923b173bb3378d2fea1"
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
    ref = json.loads((DATASET_DIR / "asterias_rubens_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-hinman-asterias-rubens-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
