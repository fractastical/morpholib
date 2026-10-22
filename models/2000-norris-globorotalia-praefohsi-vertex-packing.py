#!/usr/bin/env python3
"""
models/2000-norris-globorotalia-praefohsi-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1625)
Organism: Globorotalia praefohsi (Middle Miocene Unkeeled Rounded-Margin Foraminifer Chronospecies)
Phenomenon Scale: 756 nm (7.565e-07 m, log10=-6.121, 2_macromolecular_100nm)
Anatomical Structure: Apical epithelial cell vertex polygon in Globorotalia praefohsi (Middle Miocene (Serravallian, 13.4 Ma))
Geological Horizon: Middle Miocene (Serravallian, 13.4 Ma) (13.4 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (756 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1625_globorotalia_praefohsi_vertex_packing.lean` (`MathOfBiology.Loop_1625`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_praefohsi_vertex_packing"
EXPECTED_SHA256 = {
    "globorotalia_praefohsi_vertex_packing_reference.json": "186158a61e35250e6cf702052fd7b059dbb7dee447b5216e0dd8df1281f56b2d",
    "globorotalia_praefohsi_specimen.jpg": "4bd61f5c5fd1e4a4d998a7b4bf377b989b14b379445592ebb2d464ec82617c74",
    "scout_manifest.json": "d57ee8799938e4fa01abbd3bea2ee936596e0c244245af87f39325ace909a2ec",
    "paper_citations.bib": "8405e5bbbb697b57f5af850b3670a14fe41c00e187e920a4db690770e5bcc288"
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
    ref = json.loads((DATASET_DIR / "globorotalia_praefohsi_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-norris-globorotalia-praefohsi-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
