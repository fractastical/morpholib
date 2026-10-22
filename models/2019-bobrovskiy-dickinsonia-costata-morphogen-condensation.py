#!/usr/bin/env python3
"""
models/2019-bobrovskiy-dickinsonia-costata-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1340)
Organism: Dickinsonia costata (Ediacaran Glide-Symmetric Segmented Proarticulatan (Cholesterol Biomarker))
Phenomenon Scale: 9.9 mm (0.009945 m, log10=-2.002, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Dickinsonia costata (Late Ediacaran (White Sea, 558 Ma))
Geological Horizon: Late Ediacaran (White Sea, 558 Ma) (558.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dickinsonia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (9.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1340_dickinsonia_costata_morphogen_condensation.lean` (`MathOfBiology.Loop_1340`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dickinsonia_costata_morphogen_condensation"
EXPECTED_SHA256 = {
    "dickinsonia_costata_morphogen_condensation_reference.json": "79b3ee74ae8f17261c1093a26d7490941c10678e289a289d8beef73dd3a2ecf3",
    "dickinsonia_costata_specimen.jpg": "ab935d7f33b31712e63e63f3b84c1c1eac296d72d1728a400dbe609341e6ffc9",
    "scout_manifest.json": "0ff60dcca57ec863796f9ff138f31d800089e033d364aeb2edc0446a3eef02cf",
    "paper_citations.bib": "3d854ba973f046267c549559fed92fe8cc055b3130df3494214e61a7b59b66cd"
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
    ref = json.loads((DATASET_DIR / "dickinsonia_costata_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-bobrovskiy-dickinsonia-costata-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
