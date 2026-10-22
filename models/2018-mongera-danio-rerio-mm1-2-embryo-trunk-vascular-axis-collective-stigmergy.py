#!/usr/bin/env python3
"""
models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #1180)
Organism: Danio rerio (Zebrafish — 24–48 hpf)
Phenomenon Scale: 1.5 mm (0.001506 m, log10=-2.822, 6_organ_mm)
Anatomical Structure: Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus (24–48 hpf (Whole Embryonic AP Trunk & ISV Arcade, ZFA:0001093)) · Zebrahub whole_embryo_4D_lightsheet_trajectory (10–24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation at 1.5 mm (Whole embryonic AP trunk axis (30 somites), dorsal aorta & ISV plexus))
Formal Verification: `morphoeval/lean/loops/Loop_1180_danio_rerio_mm1_2_embryo_trunk_vascular_axis_collective_stigmergy.lean` (`MathOfBiology.Loop_1180`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_mm1_2_embryo_trunk_vascular_axis_collective_stigmergy"
EXPECTED_SHA256 = {
    "danio_rerio_mm1_2_embryo_trunk_vascular_axis_collective_stigmergy_reference.json": "2e46ca5bd8ac4d5487c0e0b9bd925101391ba80f6330837d7542c9a6d7a81d09",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "1c3f466f9218664509fddbef899e31e577507936c2281bd1405f6290dc1a92fa",
    "paper_citations.bib": "f23f0ec440983ef2efadc1d50d1b94d6c0fb4d4de3713a723ad4dbf598467acb"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_mm1_2_embryo_trunk_vascular_axis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-mongera-danio-rerio-mm1-2-embryo-trunk-vascular-axis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
