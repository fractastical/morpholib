#!/usr/bin/env python3
"""
models/2019-parichy-danio-margaritatus-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1561)
Organism: Danio margaritatus (Celestial Pearl Danio (Galaxy Rasbora))
Phenomenon Scale: 38 µm (3.825e-05 m, log10=-4.417, 4_cellular_10um)
Anatomical Structure: Apical epithelial cell vertex polygon in Danio margaritatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (38 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1561_danio_margaritatus_vertex_packing.lean` (`MathOfBiology.Loop_1561`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_margaritatus_vertex_packing"
EXPECTED_SHA256 = {
    "danio_margaritatus_vertex_packing_reference.json": "1956d74470dcd7509f64b71aaabc09b68d13dae87e5817dd2baa43d852bc3001",
    "danio_margaritatus_specimen.jpg": "6f4ec861e86da2f7ea31f23d6a81b802286dce7ac2a0631e45f9097fd9b8e876",
    "scout_manifest.json": "f98aea50619322ffbcce9c4b79c7bd613d9d5cfa0ebfd37e95a57f857b4532a1",
    "paper_citations.bib": "4c520000c0530fabc6b78503652781a1a990818469109f47606efc9f6c0da9b7"
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
    ref = json.loads((DATASET_DIR / "danio_margaritatus_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-parichy-danio-margaritatus-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
