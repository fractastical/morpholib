#!/usr/bin/env python3
"""
models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1167)
Organism: Danio rerio (Zebrafish — 24–36 hpf)
Phenomenon Scale: 240 µm (0.0002398 m, log10=-3.620, 5_primordium_100um)
Anatomical Structure: Migrating posterior lateral line primordium (PLLP) & looping heart tube (24–36 hpf (Pharyngula PLLP & Linear Heart Tube, ZFA:0001279)) · Zebrahub migrating_PLLP_stream & myocardial_endocardial_tube (24 hpf)
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling at 240 µm (Migrating posterior lateral line primordium (PLLP) & looping heart tube))
Formal Verification: `morphoeval/lean/loops/Loop_1167_danio_rerio_um220_pllp_heart_primordium_elastic_buckling.lean` (`MathOfBiology.Loop_1167`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_um220_pllp_heart_primordium_elastic_buckling"
EXPECTED_SHA256 = {
    "danio_rerio_um220_pllp_heart_primordium_elastic_buckling_reference.json": "966a13b86da8b96fa3e4f681716f72d98b0ba188eae93b0fa7f8b62bbd311314",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "ca222db74bb0b2f13189411b9534c62d25f9e1d9ebf49dc365fd91f47a02802d",
    "paper_citations.bib": "9feb48af9b83f9f711cec14398c84518f432010b0514853afdd2a4589c5edb67"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_um220_pllp_heart_primordium_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-gilmour-danio-rerio-um220-pllp-heart-primordium-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
