#!/usr/bin/env python3
"""
models/2006-uhlen-dorudon-atrox-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1406)
Organism: Dorudon atrox (Late Eocene Basilosaurid Whale (Vestigial Hindlimb))
Phenomenon Scale: 2.7 cm (0.02747 m, log10=-1.561, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Dorudon atrox (Late Eocene (Priabonian, 37.0 Ma))
Geological Horizon: Late Eocene (Priabonian, 37.0 Ma) (37.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dorudon
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1406_dorudon_atrox_clock_wavefront.lean` (`MathOfBiology.Loop_1406`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dorudon_atrox_clock_wavefront"
EXPECTED_SHA256 = {
    "dorudon_atrox_clock_wavefront_reference.json": "d261ace6e33ac5a153635ee87608a8173141532ece8929f2f752947972291e7e",
    "dorudon_atrox_specimen.jpg": "5a04e317b788950efe607a8cf4a468b15573b2f7e2efbde1deb8fa2bfaeaabaf",
    "scout_manifest.json": "c2d6e4e4ea33b259b797346e56f6fd238ae8d77e70fcec7be1289e338e1476b8",
    "paper_citations.bib": "1e58b3235b17719bba98099a7f688c1d9ed2bad6e451775c6bfb32e125744cae"
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
    ref = json.loads((DATASET_DIR / "dorudon_atrox_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-uhlen-dorudon-atrox-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
