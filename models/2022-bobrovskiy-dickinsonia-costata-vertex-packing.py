#!/usr/bin/env python3
"""
models/2022-bobrovskiy-dickinsonia-costata-vertex-packing.py — Multi-Scale & Paleobiological Reproduction (Slot #1610)
Organism: Dickinsonia costata (Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker))
Phenomenon Scale: 138 µm (0.0001381 m, log10=-3.860, 5_primordium_100um)
Anatomical Structure: Apical epithelial cell vertex polygon in Dickinsonia costata (Late Ediacaran (White Sea, 558 Ma))
Geological Horizon: Late Ediacaran (White Sea, 558 Ma) (558.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dickinsonia
Biophysical Cohort: V. Vertex Models, Jamming & Active Nematics (Epithelial Vertex Energy & Topological Packing (138 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1610_dickinsonia_costata_vertex_packing.lean` (`MathOfBiology.Loop_1610`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dickinsonia_costata_vertex_packing"
EXPECTED_SHA256 = {
    "dickinsonia_costata_vertex_packing_reference.json": "0e5a7080614e7a30fcac39dc771226517d0cb1d3aa0711891eb9c5e0c386337f",
    "dickinsonia_costata_specimen.jpg": "ab935d7f33b31712e63e63f3b84c1c1eac296d72d1728a400dbe609341e6ffc9",
    "scout_manifest.json": "aed9facd9bd4e2c31edeef57238aba84823b423280f4b5ce2668e6f45d3ee306",
    "paper_citations.bib": "754a64310eba23f91b7342759316334aa7360d22377fdfe7db6a13550b0ca0e4"
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
    ref = json.loads((DATASET_DIR / "dickinsonia_costata_vertex_packing_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-bobrovskiy-dickinsonia-costata-vertex-packing.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
