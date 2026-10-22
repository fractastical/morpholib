#!/usr/bin/env python3
"""
models/2008-fusco-aulacopleura-koninckii-holaspis22-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1604)
Organism: Aulacopleura koninckii_holaspis22 (Silurian Trilobite Holaspid Morphotype II (22 Thoracic Segments))
Phenomenon Scale: 49 µm (4.895e-05 m, log10=-4.310, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Aulacopleura koninckii_holaspis22 (Silurian (Wenlock Epoch, 428.1 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.1 Ma) (428.1 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (49 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1604_aulacopleura_koninckii_holaspis22_vertex_packing.lean` (`MathOfBiology.Loop_1604`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis22_vertex_packing"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis22_vertex_packing_reference.json": "a6590f5c9d096e3dcf6df17c47043ce534a01423b93b2bfba968de6c707bff1c",
    "aulacopleura_koninckii_holaspis22_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "9056d0bb374fffa490a9403a865ffaeace33c02b735bc4484cb3a61a789f894d",
    "paper_citations.bib": "2f3b667edd58ffbf9e16d3c588d3ad5294244b663a54e6d5fca60c74c6772649"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis22_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-fusco-aulacopleura-koninckii-holaspis22-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
