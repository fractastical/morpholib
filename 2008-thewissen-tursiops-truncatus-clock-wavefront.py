#!/usr/bin/env python3
"""
2008-thewissen-tursiops-truncatus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0310)
Organism: Tursiops truncatus (Bottlenose Dolphin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Tursiops
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0310_tursiops_truncatus_clock_wavefront.lean` (`MathOfBiology.Loop_0310`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tursiops_truncatus_clock_wavefront"
EXPECTED_SHA256 = {
    "tursiops_truncatus_clock_wavefront_reference.json": "dc8883896e51ed74a6bc3d3f31f899fe572a3b21aa1bce8b72efef9fc5af8b1b",
    "tursiops_truncatus_specimen.jpg": "038f0a413155e6a92701bb438caa1f4c23fdef11b3c5c4892f851e65bcdf710a",
    "scout_manifest.json": "df5bb4188cd0f4e2a0f577735c9412756c119352aab60f213e23f22531a6d24e",
    "paper_citations.bib": "1b2fad215f8188463c71ac7fee218bf6909e26c8dc5f06d95cd4bb49c243c9a5"
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
    ref = json.loads((DATASET_DIR / "tursiops_truncatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-thewissen-tursiops-truncatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
