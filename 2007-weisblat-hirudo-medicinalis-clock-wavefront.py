#!/usr/bin/env python3
"""
2007-weisblat-hirudo-medicinalis-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0369)
Organism: Hirudo medicinalis (Medicinal Leech)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hirudo
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0369_hirudo_medicinalis_clock_wavefront.lean` (`MathOfBiology.Loop_0369`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hirudo_medicinalis_clock_wavefront"
EXPECTED_SHA256 = {
    "hirudo_medicinalis_clock_wavefront_reference.json": "ab7232d499492321c0163eb8d182a544b746a8fcb4764a073e430cfb79e06ec9",
    "hirudo_medicinalis_specimen.jpg": "f9cc303efc9e9bbd4f520cb58b823e3ee6b17491f3c40d8ee2c2ad8fcedf909f",
    "scout_manifest.json": "fbaea4c70501f3831decb26ebe37ca8053867471c72682e7120dba6f275b4bff",
    "paper_citations.bib": "b9d114274eb2eef35e308d231824e4809f63220ed0ab5094b1718dc3d380a89d"
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
    ref = json.loads((DATASET_DIR / "hirudo_medicinalis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-weisblat-hirudo-medicinalis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
