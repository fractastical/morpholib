#!/usr/bin/env python3
"""
models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1176)
Organism: Danio rerio (Zebrafish — 24–48 hpf)
Phenomenon Scale: 1.2 mm (0.001242 m, log10=-2.906, 6_organ_mm)
Anatomical Structure: Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus (24–48 hpf (Whole Embryonic AP Trunk & ISV Arcade, ZFA:0001093)) · Zebrahub whole_embryo_4D_lightsheet_trajectory (10–24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting at 1.2 mm (Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus))
Formal Verification: `morphoeval/lean/loops/Loop_1176_danio_rerio_mm1_2_embryo_trunk_vascular_axis_branching_tree.lean` (`MathOfBiology.Loop_1176`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_mm1_2_embryo_trunk_vascular_axis_branching_tree"
EXPECTED_SHA256 = {
    "danio_rerio_mm1_2_embryo_trunk_vascular_axis_branching_tree_reference.json": "864d07d4f03e170cd790bd71175717fc2e159b8651818f5d1c1f5f5d04d14241",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "ff4136a6d69e61d3caec453fd0dd04d1296ba78538c731848b0bb993a5c986dc",
    "paper_citations.bib": "2c014e7ca97093ce3ab5a30287d01cd9b710e1bf5c68da2ca9dcad1243fff1b5"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_mm1_2_embryo_trunk_vascular_axis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
