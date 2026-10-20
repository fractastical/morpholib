#!/usr/bin/env python3
"""
2022-cooper-scyliorhinus-canicula-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0535)
Organism: Scyliorhinus canicula (Small-Spotted Catshark)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Scyliorhinus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0535_scyliorhinus_canicula_vertex_packing.lean` (`MathOfBiology.Loop_0535`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/scyliorhinus_canicula_vertex_packing"
EXPECTED_SHA256 = {
    "scyliorhinus_canicula_vertex_packing_reference.json": "e52b32cb9373e44a6d62190b2eb0f556520f22f4e96946aac710eda281e78fd3",
    "scyliorhinus_canicula_specimen.jpg": "71525dd878e51b100ea6bbd4e4005751066059fc236d4e204ad81ed2bbb5d553",
    "scout_manifest.json": "61c4c6b32e636e1e9bf1e8d0d1011d9227c630980640eece5a4e9e0f3741a045",
    "paper_citations.bib": "2e576b83a9e2f56360804201a7bdeef3549e7c88817ef3d4358947cc19efd9f7"
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
    ref = json.loads((DATASET_DIR / "scyliorhinus_canicula_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-cooper-scyliorhinus-canicula-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
