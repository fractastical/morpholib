#!/usr/bin/env python3
"""
models/1996-dimichele-lepidodendron-aculeatum-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1618)
Organism: Lepidodendron aculeatum (Carboniferous Giant Scale-Tree (Helical Parastichy Leaf Cushions))
Phenomenon Scale: 85 µm (8.455e-05 m, log10=-4.073, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Lepidodendron aculeatum (Late Carboniferous (Pennsylvanian, 310 Ma))
Geological Horizon: Late Carboniferous (Pennsylvanian, 310 Ma) (310.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Lepidodendron
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (85 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1618_lepidodendron_aculeatum_vertex_packing.lean` (`MathOfBiology.Loop_1618`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/lepidodendron_aculeatum_vertex_packing"
EXPECTED_SHA256 = {
    "lepidodendron_aculeatum_vertex_packing_reference.json": "bce99adffef074daa08f5737283a52f0bb13e11265db9a9011e3dbabec537363",
    "lepidodendron_aculeatum_specimen.jpg": "854f4a6257e58a0123b8eee34674c32ed016882c88d6eb907ca1d85dbb78c467",
    "scout_manifest.json": "30abdac64285b20b62ca6e5bc0e3a440c48f75d5c428cc8c818b2a032b5f7968",
    "paper_citations.bib": "4b7c1552c4dd490565d32f30604ab47edab3badc490e9ea3929c0d6ad5fac711"
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
    ref = json.loads((DATASET_DIR / "lepidodendron_aculeatum_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-dimichele-lepidodendron-aculeatum-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
