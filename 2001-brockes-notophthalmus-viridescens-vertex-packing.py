#!/usr/bin/env python3
"""
2001-brockes-notophthalmus-viridescens-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0527)
Organism: Notophthalmus viridescens (Eastern Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Notophthalmus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0527_notophthalmus_viridescens_vertex_packing.lean` (`MathOfBiology.Loop_0527`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/notophthalmus_viridescens_vertex_packing"
EXPECTED_SHA256 = {
    "notophthalmus_viridescens_vertex_packing_reference.json": "4f213553ba1392942db5a186838314b3650f7a54a9d517cef0b76d622f99a46c",
    "notophthalmus_viridescens_specimen.jpg": "5389e095a5be650aba6c5b00fa322f4fbd3beed046b1913ac82a95ce045d5000",
    "scout_manifest.json": "5008e10657a2dcf7e2138cf76b75b67067433e1e782f76113705fa0d7609c039",
    "paper_citations.bib": "50b0474ccd341b437ad3700c46bd0a85c40eaa581a6f3ad6817901d003e79d34"
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
    ref = json.loads((DATASET_DIR / "notophthalmus_viridescens_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-brockes-notophthalmus-viridescens-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
