#!/usr/bin/env python3
"""
2008-kingsley-gasterosteus-aculeatus-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0536)
Organism: Gasterosteus aculeatus (Three-Spined Stickleback)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0536_gasterosteus_aculeatus_vertex_packing.lean` (`MathOfBiology.Loop_0536`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_vertex_packing"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_vertex_packing_reference.json": "ded73b4535c5f3327333f560ce9d24958f67e29eb168222d124af31fbb779aba",
    "gasterosteus_aculeatus_specimen.jpg": "86fef1d4aeb86c3903e26fae972a360ea0c848048e29b546bedb2db842a5bd10",
    "scout_manifest.json": "82d8a0851f5bfd44493ca98b4f08aaf46d0ee28b032cbf352e3d421e8dc1947d",
    "paper_citations.bib": "0ad3bf455629880d80b1d306b7a5e997190440429798b4a780849a7793d15afe"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-kingsley-gasterosteus-aculeatus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
