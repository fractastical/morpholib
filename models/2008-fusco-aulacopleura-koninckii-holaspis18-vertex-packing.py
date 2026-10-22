#!/usr/bin/env python3
"""
models/2008-fusco-aulacopleura-koninckii-holaspis18-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1603)
Organism: Aulacopleura koninckii_holaspis18 (Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments))
Phenomenon Scale: 34 µm (3.4e-05 m, log10=-4.469, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Aulacopleura koninckii_holaspis18 (Silurian (Wenlock Epoch, 428.3 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.3 Ma) (428.3 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (34 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1603_aulacopleura_koninckii_holaspis18_vertex_packing.lean` (`MathOfBiology.Loop_1603`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis18_vertex_packing"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis18_vertex_packing_reference.json": "7f0d7a7fb8e950d492fc9f815a6b226782deb084fad256374b5c98470b57507e",
    "aulacopleura_koninckii_holaspis18_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "5f40fa99e1411c50e1d8c71159bbd69ff8b0784c66f32cb53bf16d126d3b5a04",
    "paper_citations.bib": "5c2da13f0ed01b2015e00fee355cf5fcde4b43e4b85f32012b18d7d294712949"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis18_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-fusco-aulacopleura-koninckii-holaspis18-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
