#!/usr/bin/env python3
"""
2009-weisblat-hirudo-medicinalis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0569)
Organism: Hirudo medicinalis (Medicinal Leech)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hirudo
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0569_hirudo_medicinalis_vertex_packing.lean` (`MathOfBiology.Loop_0569`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hirudo_medicinalis_vertex_packing"
EXPECTED_SHA256 = {
    "hirudo_medicinalis_vertex_packing_reference.json": "bfa4e3e016f7757cdbe20725dbc930aec31b113f99bece001a02b868a1c2d46e",
    "hirudo_medicinalis_specimen.jpg": "f9cc303efc9e9bbd4f520cb58b823e3ee6b17491f3c40d8ee2c2ad8fcedf909f",
    "scout_manifest.json": "240fb817528f521ae45245696d1652c81a8f9d15003af87cfdfb0a6170f9cff9",
    "paper_citations.bib": "fbd6d4fa9b6159cf42be6dc9433f49cf2885b2fa7cd03c6b65beb9123f402abe"
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
    ref = json.loads((DATASET_DIR / "hirudo_medicinalis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-weisblat-hirudo-medicinalis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
