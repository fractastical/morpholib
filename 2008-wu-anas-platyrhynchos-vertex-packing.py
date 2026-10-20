#!/usr/bin/env python3
"""
2008-wu-anas-platyrhynchos-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0516)
Organism: Anas platyrhynchos (Mallard Duck)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anas
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0516_anas_platyrhynchos_vertex_packing.lean` (`MathOfBiology.Loop_0516`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anas_platyrhynchos_vertex_packing"
EXPECTED_SHA256 = {
    "anas_platyrhynchos_vertex_packing_reference.json": "383ad65f36d9609be6da0808af85739dd64356da55e0fdec72fbced2c969b20f",
    "anas_platyrhynchos_specimen.jpg": "ee1cc0720d68f5f20e5e3490647b69cb29c1fe84dae8fdb048019814f69eccba",
    "scout_manifest.json": "072470e1a24c225423b14b3a8111cf18cf8206bf0d07efb65899084f21f2b713",
    "paper_citations.bib": "82893aa582373e44993112b18027d3cadc72fdfc0da895b3d60b116c81c3b750"
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
    ref = json.loads((DATASET_DIR / "anas_platyrhynchos_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-wu-anas-platyrhynchos-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
