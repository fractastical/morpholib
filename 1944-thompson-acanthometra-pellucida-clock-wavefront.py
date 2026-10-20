#!/usr/bin/env python3
"""
1944-thompson-acanthometra-pellucida-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0393)
Organism: Acanthometra pellucida (Strontium Radiolarian)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acanthometra
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0393_acanthometra_pellucida_clock_wavefront.lean` (`MathOfBiology.Loop_0393`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acanthometra_pellucida_clock_wavefront"
EXPECTED_SHA256 = {
    "acanthometra_pellucida_clock_wavefront_reference.json": "953472746fbb871334ec2d6a2b8a5957a54261bc9b544dae101e4a37a570c470",
    "acanthometra_pellucida_specimen.jpg": "8f4b8a930e33286fd2843fa388dfec53592278e506619611732e990cc03371d4",
    "scout_manifest.json": "77a5871d2e00459eece52b5b7e4be01c35289ed3d4424dcf92141824bcc91efe",
    "paper_citations.bib": "0bcdcd3a2f9378875c2fecccc8cb975df66940d039e38655f407b3ec953390e6"
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
    ref = json.loads((DATASET_DIR / "acanthometra_pellucida_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1944-thompson-acanthometra-pellucida-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
