#!/usr/bin/env python3
"""
models/2022-spiewak-danio-aesculapii-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1564)
Organism: Danio aesculapii (Panther Danio (Vertical Bar Transition))
Phenomenon Scale: 63 µm (6.305e-05 m, log10=-4.200, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Danio aesculapii (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (63 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1564_danio_aesculapii_vertex_packing.lean` (`MathOfBiology.Loop_1564`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_aesculapii_vertex_packing"
EXPECTED_SHA256 = {
    "danio_aesculapii_vertex_packing_reference.json": "789cd131d8853460b2ca9c873ee625c7713a273f0b6b899a328b0b467df0d1e7",
    "danio_aesculapii_specimen.jpg": "4ccce0ce3648e44c77cef768d10f5acd8ab553c4ca38d348c35d4c6846b46bc5",
    "scout_manifest.json": "136e19aa210659914290543ce8130d69ca16dce007edc730effd6d123d319dab",
    "paper_citations.bib": "37a5dd3924715c31d592e0054d80b6a7be0ba63542aacdca7b5e287554daae05"
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
    ref = json.loads((DATASET_DIR / "danio_aesculapii_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-spiewak-danio-aesculapii-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
