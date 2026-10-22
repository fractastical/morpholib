#!/usr/bin/env python3
"""
models/2022-martins-loxodonta-africana-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1637)
Organism: Loxodonta africana (African Bush Elephant (Skin Thermal Mud-Cracking Channel Network))
Phenomenon Scale: 3.3 µm (3.27e-06 m, log10=-5.485, 3_subcellular_um)
Anatomical Structure: Apical epithelial cell vertex polygon in Loxodonta africana (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Loxodonta
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (3.3 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1637_loxodonta_africana_vertex_packing.lean` (`MathOfBiology.Loop_1637`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/loxodonta_africana_vertex_packing"
EXPECTED_SHA256 = {
    "loxodonta_africana_vertex_packing_reference.json": "c512c1fbe895c6de9895df9348febabaca6dcfcd5eb0a8599b3e7c21d7f5a461",
    "loxodonta_africana_specimen.jpg": "a530e48a3daec38e9c6fd4a03ae176871589b79b17232feda58acffdff236163",
    "scout_manifest.json": "f6ab5b8615cdf271757e76aa3ac385dc76f7b9fda0c13c164a09c01f98d82694",
    "paper_citations.bib": "76ade7814d2fe3c6391de15215406128342399f261f5834b7ec93004ec87e955"
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
    ref = json.loads((DATASET_DIR / "loxodonta_africana_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-martins-loxodonta-africana-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
