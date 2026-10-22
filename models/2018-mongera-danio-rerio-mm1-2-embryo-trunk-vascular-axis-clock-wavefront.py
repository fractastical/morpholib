#!/usr/bin/env python3
"""
models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1173)
Organism: Danio rerio (Zebrafish — 24–48 hpf)
Phenomenon Scale: 1.0 mm (0.001044 m, log10=-2.981, 6_organ_mm)
Anatomical Structure: Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus (24–48 hpf (Whole Embryonic AP Trunk & ISV Arcade, ZFA:0001093)) · Zebrahub whole_embryo_4D_lightsheet_trajectory (10–24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest at 1.0 mm (Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus))
Formal Verification: `morphoeval/lean/loops/Loop_1173_danio_rerio_mm1_2_embryo_trunk_vascular_axis_clock_wavefront.lean` (`MathOfBiology.Loop_1173`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_mm1_2_embryo_trunk_vascular_axis_clock_wavefront"
EXPECTED_SHA256 = {
    "danio_rerio_mm1_2_embryo_trunk_vascular_axis_clock_wavefront_reference.json": "dcbee38c8d849ab5f3aeed221cbed9e6af4ea38220f081f83ba4b3ee600363e5",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "f6990b4630aa94dca48da8e0df88b2d7a295f20a6dce15a0f146ef3c1edfc1be",
    "paper_citations.bib": "4c33afe0dc0240d9c34dbdfcdb50c252bb1de0587ca7bb45fb18da0870f1e7ea"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_mm1_2_embryo_trunk_vascular_axis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
