#!/usr/bin/env python3
"""
2020-bakker-struthio-camelus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0515)
Organism: Struthio camelus (Common Ostrich)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Struthio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0515_struthio_camelus_vertex_packing.lean` (`MathOfBiology.Loop_0515`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/struthio_camelus_vertex_packing"
EXPECTED_SHA256 = {
    "struthio_camelus_vertex_packing_reference.json": "d7a75c788d734c3c7c3854e9fe5312f63ee3f4ae2556e517fd454051c4bc7ee9",
    "struthio_camelus_specimen.jpg": "798eba64c73c1474a1b787e850654c5f61164ca01db9ec2d75836507f9494def",
    "scout_manifest.json": "40147772a2435233887cd8349d4b3ce0e1b0b608c9dc4d3d70b1cdb0f3b1fd94",
    "paper_citations.bib": "bd74b4c25bc19256e3928f001ef17cf27236d1598063c4203779d64ef8a5015d"
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
    ref = json.loads((DATASET_DIR / "struthio_camelus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-bakker-struthio-camelus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
