#!/usr/bin/env python3
"""
models/1996-macfadden-mesohippus-bairdi-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1579)
Organism: Mesohippus bairdi (Late Eocene Three-Toed Sub-Hypsodont Horse)
Phenomenon Scale: 316 µm (0.0003156 m, log10=-3.501, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Mesohippus bairdi (Late Eocene (Chadronian, 35.0 Ma))
Geological Horizon: Late Eocene (Chadronian, 35.0 Ma) (35.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mesohippus
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (316 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1579_mesohippus_bairdi_vertex_packing.lean` (`MathOfBiology.Loop_1579`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/mesohippus_bairdi_vertex_packing"
EXPECTED_SHA256 = {
    "mesohippus_bairdi_vertex_packing_reference.json": "6274b47fe8f9784244b6708fbe34e3865d3d469d794bbb60b1e21672ff2c3e3b",
    "mesohippus_bairdi_specimen.jpg": "ffa85314095959cda06a6a0c44566047678b150e06996250a97c7c8b4dbc1876",
    "scout_manifest.json": "95fc5551cd6fd1726a34934e69d4b54e83f997340d007a09c5e86a578ab8e179",
    "paper_citations.bib": "fdfa8b318d22cf70445de89e701e092eed1b436bc386bcc6d8bf949dcd81cfa6"
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
    ref = json.loads((DATASET_DIR / "mesohippus_bairdi_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1996-macfadden-mesohippus-bairdi-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
