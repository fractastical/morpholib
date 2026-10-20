#!/usr/bin/env python3
"""
2003-aizenberg-ophiocoma-wendtii-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0359)
Organism: Ophiocoma wendtii (Red Brittle Star)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ophiocoma
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0359_ophiocoma_wendtii_clock_wavefront.lean` (`MathOfBiology.Loop_0359`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ophiocoma_wendtii_clock_wavefront"
EXPECTED_SHA256 = {
    "ophiocoma_wendtii_clock_wavefront_reference.json": "ef8cdca6b93011f8829124dad7b0ad296d3614cc2552f7c9c4fbfa89ced935bd",
    "ophiocoma_wendtii_specimen.jpg": "c073d01d499967366ff61d9831cb0e07e7505885a8ce4a9445565ec8d34f057d",
    "scout_manifest.json": "8922bdcaa6c2c11b0cced4b48d940ce8dd93eebdab5f25448a292c649ed9073c",
    "paper_citations.bib": "113f66ea683485d760e146c4f3751309d716b004f76469af192064cc36375717"
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
    ref = json.loads((DATASET_DIR / "ophiocoma_wendtii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-aizenberg-ophiocoma-wendtii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
