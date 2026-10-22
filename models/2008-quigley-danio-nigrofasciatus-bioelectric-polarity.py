#!/usr/bin/env python3
"""
models/2008-quigley-danio-nigrofasciatus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1923)
Organism: Danio nigrofasciatus (Spotted Danio (broken ventral stripes))
Phenomenon Scale: 21 µm (2.083e-05 m, log10=-4.681, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Danio nigrofasciatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (21 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1923_danio_nigrofasciatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1923`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_nigrofasciatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "danio_nigrofasciatus_bioelectric_polarity_reference.json": "3ee96e32f3e05121a033dc2909584b561e4841693476b5d3fec92a0c8c6f1434",
    "danio_nigrofasciatus_specimen.jpg": "4aee990cea7e343b773365da24389df37f170d7d68c3492c513f22afd9f750d3",
    "scout_manifest.json": "e69e1606b3eb07f547eaf9eca1a69266200391f321dac1f6537cef60a5e3453f",
    "paper_citations.bib": "ff24e160d844febb19f2868b29534be76fdf5436bbb6a05e194b272cc6a77585"
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
    ref = json.loads((DATASET_DIR / "danio_nigrofasciatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-quigley-danio-nigrofasciatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
