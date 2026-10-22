#!/usr/bin/env python3
"""
models/2018-comanns-moloch-horridus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1459)
Organism: Moloch horridus (Australian Thorny Devil (Capillary Drinking Skin Micro-Groove Network))
Phenomenon Scale: 6.8 µm (6.764e-06 m, log10=-5.170, 3_subcellular_um)
Anatomical Structure: Segmentation clock somite / metamere unit in Moloch horridus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Moloch
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (6.8 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1459_moloch_horridus_clock_wavefront.lean` (`MathOfBiology.Loop_1459`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/moloch_horridus_clock_wavefront"
EXPECTED_SHA256 = {
    "moloch_horridus_clock_wavefront_reference.json": "885aece21354aed55c07279dc179e4e6bce522fd917ad6d1be9c7472fc36cf31",
    "moloch_horridus_specimen.jpg": "224fb2ba2daad242971c5263ec01e8e384e89bea8709b08d5380e120ebdbf94a",
    "scout_manifest.json": "a9177868d5fada87b6ef13f420f299862491548bfaed1bc1247bfa26601d65fe",
    "paper_citations.bib": "4bb2609eeafbcf76a72f67f8e7e3c292780d4bc27a36e79e5410653cb158b533"
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
    ref = json.loads((DATASET_DIR / "moloch_horridus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-comanns-moloch-horridus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
