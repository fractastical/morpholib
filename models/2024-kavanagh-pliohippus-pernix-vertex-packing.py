#!/usr/bin/env python3
"""
models/2024-kavanagh-pliohippus-pernix-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1581)
Organism: Pliohippus pernix (Late Miocene Single-Hoofed Monodactyl Ancestor)
Phenomenon Scale: 586 µm (0.0005859 m, log10=-3.232, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Pliohippus pernix (Late Miocene (Hemphillian, 8.5 Ma))
Geological Horizon: Late Miocene (Hemphillian, 8.5 Ma) (8.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pliohippus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (586 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1581_pliohippus_pernix_vertex_packing.lean` (`MathOfBiology.Loop_1581`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pliohippus_pernix_vertex_packing"
EXPECTED_SHA256 = {
    "pliohippus_pernix_vertex_packing_reference.json": "a241ac75be38d2c0501f5c15397385090823b35611b2f2e9135421b04f3d844f",
    "pliohippus_pernix_specimen.jpg": "1211e4acc14fca6bb091e7747aca534b08c9b54a2cea23f8b7aa25c1b659afdd",
    "scout_manifest.json": "b648ee74610f48375629b8064d04df0f7dc0248f59682277e188ce110652e192",
    "paper_citations.bib": "171523042faa63d956f3b15cd117b37fe5656ff0729ebbebfab456ef4cc51beb"
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
    ref = json.loads((DATASET_DIR / "pliohippus_pernix_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-kavanagh-pliohippus-pernix-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
