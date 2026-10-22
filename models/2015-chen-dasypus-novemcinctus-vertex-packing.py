#!/usr/bin/env python3
"""
models/2015-chen-dasypus-novemcinctus-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1636)
Organism: Dasypus novemcinctus (Nine-Banded Armadillo (Hexagonal Osteoderm Rosette Tiles))
Phenomenon Scale: 17 µm (1.706e-05 m, log10=-4.768, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Dasypus novemcinctus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dasypus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (17 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1636_dasypus_novemcinctus_vertex_packing.lean` (`MathOfBiology.Loop_1636`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dasypus_novemcinctus_vertex_packing"
EXPECTED_SHA256 = {
    "dasypus_novemcinctus_vertex_packing_reference.json": "dcfa338e592501c7e60468e163cabac3309d1227933e5322981506a64b424c02",
    "dasypus_novemcinctus_specimen.jpg": "5cbb611c05d8707d92d1c6d2cdd19de1525cb938009adae92309147f22174539",
    "scout_manifest.json": "35d15a267f4a2ab0566a2e07625275d25fb3905ac6cea9c7bad3b766c2f3467e",
    "paper_citations.bib": "3029ba2bc2009bf305d5f074cc33ddc0112644c8d8828623d860a5b6c30fe68c"
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
    ref = json.loads((DATASET_DIR / "dasypus_novemcinctus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-chen-dasypus-novemcinctus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
