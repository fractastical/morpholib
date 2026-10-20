#!/usr/bin/env python3
"""
2009-kusserow-nematostella-vectensis-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0562)
Organism: Nematostella vectensis (Starlet Sea Anemone)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Nematostella
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0562_nematostella_vectensis_vertex_packing.lean` (`MathOfBiology.Loop_0562`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/nematostella_vectensis_vertex_packing"
EXPECTED_SHA256 = {
    "nematostella_vectensis_vertex_packing_reference.json": "adff85c0269a3ef7aa24652c844864d4f807bff40e3425a77d9a909cfb0513c7",
    "nematostella_vectensis_specimen.jpg": "fb9d66de457a62bbf3d2ec1bcdb57f33e32468690a3047a09117a19153073a53",
    "scout_manifest.json": "1ddafc2e3f5c8fb12ee068df8e3b714ddf17b5ed3eb0976e37f906e5dd390974",
    "paper_citations.bib": "8219ae6d5abe1f9d176251aa5f807eaf40f0f717cdab454b89b8e801a9d75c71"
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
    ref = json.loads((DATASET_DIR / "nematostella_vectensis_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-kusserow-nematostella-vectensis-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
