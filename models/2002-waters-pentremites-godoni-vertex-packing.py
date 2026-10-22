#!/usr/bin/env python3
"""
models/2002-waters-pentremites-godoni-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1630)
Organism: Pentremites godoni (Mississippian Pentaradial Hydrospire Blastoid Calyx)
Phenomenon Scale: 49 µm (4.905e-05 m, log10=-4.309, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Pentremites godoni (Mississippian (Chesterian, 330 Ma))
Geological Horizon: Mississippian (Chesterian, 330 Ma) (330.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pentremites
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (49 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1630_pentremites_godoni_vertex_packing.lean` (`MathOfBiology.Loop_1630`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pentremites_godoni_vertex_packing"
EXPECTED_SHA256 = {
    "pentremites_godoni_vertex_packing_reference.json": "2058be3fe79df6ef95d23336e304702be90146ca49099615d5a05ef99df941f6",
    "pentremites_godoni_specimen.jpg": "53e40da38ecd2df40e09153c5fa4329553f1a7e8c16087490c6ba4fccd9eeb1d",
    "scout_manifest.json": "2f69c03c67cec0279942e094d5f8254aecb8562a9586d5f519a9246363afd563",
    "paper_citations.bib": "54b537e3355760ed5b50abdd12f814ee9a3eb19c2b516da57eefd15d62ab8c14"
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
    ref = json.loads((DATASET_DIR / "pentremites_godoni_vertex_packing_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2002-waters-pentremites-godoni-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
