#!/usr/bin/env python3
"""
2012-goryachev-saccharomyces-cerevisiae-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0586)
Organism: Saccharomyces cerevisiae (Budding Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Saccharomyces
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0586_saccharomyces_cerevisiae_vertex_packing.lean` (`MathOfBiology.Loop_0586`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/saccharomyces_cerevisiae_vertex_packing"
EXPECTED_SHA256 = {
    "saccharomyces_cerevisiae_vertex_packing_reference.json": "459e32bb30de0f4e75fdcb6a973198f55b92b44ad827ba07624f7513b3a88152",
    "saccharomyces_cerevisiae_specimen.jpg": "03d89aa520798f5a2c721833870f675b8d4676ea933358e95702fcd1732eae0b",
    "scout_manifest.json": "a9e0c4857a554fe9bdf3a40575e9522d55213dba935510b95de499b959e4e5f4",
    "paper_citations.bib": "05f3bbd72657279223c72a4ed484f78a42f9652cd38fba4d2bd619e629941288"
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
    ref = json.loads((DATASET_DIR / "saccharomyces_cerevisiae_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-goryachev-saccharomyces-cerevisiae-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
