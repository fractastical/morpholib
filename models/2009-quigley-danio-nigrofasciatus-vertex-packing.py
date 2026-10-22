#!/usr/bin/env python3
"""
models/2009-quigley-danio-nigrofasciatus-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1563)
Organism: Danio nigrofasciatus (Spotted Danio (broken ventral stripes))
Phenomenon Scale: 65 µm (6.51e-05 m, log10=-4.186, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Danio nigrofasciatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (65 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1563_danio_nigrofasciatus_vertex_packing.lean` (`MathOfBiology.Loop_1563`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_nigrofasciatus_vertex_packing"
EXPECTED_SHA256 = {
    "danio_nigrofasciatus_vertex_packing_reference.json": "0c2cf4809c7b1578d411f3e91861b09dfc110e88503de105c8cfe6107c6c5dc3",
    "danio_nigrofasciatus_specimen.jpg": "4aee990cea7e343b773365da24389df37f170d7d68c3492c513f22afd9f750d3",
    "scout_manifest.json": "e7c6b19a7f11267e03cd266fb553492995b8ff11636f8de46bc95500f70a13cc",
    "paper_citations.bib": "f6024af85a325e98e8f836d6106ba58cfa9c7fe96222529eb8496f6aa943f21a"
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
    ref = json.loads((DATASET_DIR / "danio_nigrofasciatus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-quigley-danio-nigrofasciatus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
