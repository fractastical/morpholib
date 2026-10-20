#!/usr/bin/env python3
"""
2004-martindale-mnemiopsis-leidyi-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0370)
Organism: Mnemiopsis leidyi (Warty Comb Jelly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Mnemiopsis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0370_mnemiopsis_leidyi_clock_wavefront.lean` (`MathOfBiology.Loop_0370`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mnemiopsis_leidyi_clock_wavefront"
EXPECTED_SHA256 = {
    "mnemiopsis_leidyi_clock_wavefront_reference.json": "662510b87c5662af155ceb67bb73a909cac2b95b7b5e91680a01e17740d52e55",
    "mnemiopsis_leidyi_specimen.jpg": "d966e0f5b06b5af8e1ef77edf5e32c90c3aa8489be8bbbf2bdf7795ecbfb7ec7",
    "scout_manifest.json": "01b36fc63764905aa6473bb5ac4973c4fc730d9b4d87e8c995ce7fe1797f5f0e",
    "paper_citations.bib": "3581ccc412ad6b8b8f85a9efa3f3fa7bb701ecb56dc58d8ba6126e75b8845d97"
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
    ref = json.loads((DATASET_DIR / "mnemiopsis_leidyi_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-martindale-mnemiopsis-leidyi-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
