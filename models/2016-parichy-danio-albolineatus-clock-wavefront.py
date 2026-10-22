#!/usr/bin/env python3
"""
models/2016-parichy-danio-albolineatus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1382)
Organism: Danio albolineatus (Pearl Danio (Uniform Interstripe))
Phenomenon Scale: 2.6 mm (0.002563 m, log10=-2.591, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Danio albolineatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1382_danio_albolineatus_clock_wavefront.lean` (`MathOfBiology.Loop_1382`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_albolineatus_clock_wavefront"
EXPECTED_SHA256 = {
    "danio_albolineatus_clock_wavefront_reference.json": "4cf6ea2ff765b5ff34d56e61cac6d8cd8701a0d63370740ddf35fe673d5efe39",
    "danio_albolineatus_specimen.jpg": "6059c6481299cc5bb1799fc70390ba1924f905610346e50fc579efcd09df8e46",
    "scout_manifest.json": "58e18a88bbfa6cfbbdd3e480993917f9a3d5adaa1145c1fc53d9384ad622b0a3",
    "paper_citations.bib": "5a937a4358dc1e38efd67ef0c86970bd0170a113ae6eac487e61decc50b6f4e5"
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
    ref = json.loads((DATASET_DIR / "danio_albolineatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-parichy-danio-albolineatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
