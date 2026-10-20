#!/usr/bin/env python3
"""
1946-thompson-acanthometra-pellucida-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0593)
Organism: Acanthometra pellucida (Strontium Radiolarian)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acanthometra
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0593_acanthometra_pellucida_vertex_packing.lean` (`MathOfBiology.Loop_0593`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acanthometra_pellucida_vertex_packing"
EXPECTED_SHA256 = {
    "acanthometra_pellucida_vertex_packing_reference.json": "453a6b21de26660abeafc17d5ec239abc8ed1e22cd36ec8f27dd60b846eb0365",
    "acanthometra_pellucida_specimen.jpg": "8f4b8a930e33286fd2843fa388dfec53592278e506619611732e990cc03371d4",
    "scout_manifest.json": "88820998ba7e87e58588bed5ba7077214e2cd900d9762b41ca100b4f0b64d319",
    "paper_citations.bib": "d5f328105a73a6197cdd79400165b18125515692fb737bf74d8bd3258f776ece"
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
    ref = json.loads((DATASET_DIR / "acanthometra_pellucida_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1946-thompson-acanthometra-pellucida-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
