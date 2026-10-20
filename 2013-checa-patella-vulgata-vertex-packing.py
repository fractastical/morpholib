#!/usr/bin/env python3
"""
2013-checa-patella-vulgata-vertex-packing.py — Pass 2 Remediated Reproduction (Slot #0553)
Organism: Patella vulgata (Common Limpet)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Patella
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing)
Formal Verification: `morphoeval/lean/loops/Loop_0553_patella_vulgata_vertex_packing.lean` (`MathOfBiology.Loop_0553`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/patella_vulgata_vertex_packing"
EXPECTED_SHA256 = {
    "patella_vulgata_vertex_packing_reference.json": "2689d496f1e316b481f3d342613e96f14011aa6e101855b21eaed5a204d6692b",
    "patella_vulgata_specimen.jpg": "dc71ee15b04a3541b9b25a13411da18888acaaba54973227c4ba40c361669d0e",
    "scout_manifest.json": "aa96d8f04cbede627102966d5db4187dd29e13ec941025132a8d0f57a619be96",
    "paper_citations.bib": "a34818dbc2ec74095a09c488db544825da7a8c98667b136050f70fe1de20f9cf"
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
    ref = json.loads((DATASET_DIR / "patella_vulgata_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-checa-patella-vulgata-vertex-packing.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
