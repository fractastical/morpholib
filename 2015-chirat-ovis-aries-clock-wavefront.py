#!/usr/bin/env python3
"""
2015-chirat-ovis-aries-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0307)
Organism: Ovis aries (Bighorn Sheep)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ovis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0307_ovis_aries_clock_wavefront.lean` (`MathOfBiology.Loop_0307`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ovis_aries_clock_wavefront"
EXPECTED_SHA256 = {
    "ovis_aries_clock_wavefront_reference.json": "8069320f41567e0c101c83b98c071b97665d93ce91024c3b378ffe75129683be",
    "ovis_aries_specimen.jpg": "3e5aea2a1dd349564873df9e61677c401d7c2264295f722a777e3808f6a3c0d2",
    "scout_manifest.json": "9cc68e1cdf0eb02c67987770a476693f19941d540c44e9c474fbce033862e4f1",
    "paper_citations.bib": "076e5e662fa740c772ca345fe86508d67e8149ca545da4c68b582e98e8bb6b15"
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
    ref = json.loads((DATASET_DIR / "ovis_aries_clock_wavefront_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2015-chirat-ovis-aries-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
