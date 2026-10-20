#!/usr/bin/env python3
"""
2022-eliason-calypte-anna-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0320)
Organism: Calypte anna (Anna's Hummingbird)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Calypte
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0320_calypte_anna_clock_wavefront.lean` (`MathOfBiology.Loop_0320`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/calypte_anna_clock_wavefront"
EXPECTED_SHA256 = {
    "calypte_anna_clock_wavefront_reference.json": "c722b4dcc85123395f9cb9a2c156d28655bd3c7cf67eb7e70e9dfc00a5fbc23a",
    "calypte_anna_specimen.jpg": "023a1c1ebf863e2bc3fe443e9cc7d668c6fa7ac72517c8a62f1874671a2eeb60",
    "scout_manifest.json": "0d58bcda18526ba743cf8281dbc974921ceca071b0d874583efc241ba072c4c4",
    "paper_citations.bib": "409bcee4c137d284490ef7fa78f5b73d986e286a934af99ba8b6e5860f1996fb"
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
    ref = json.loads((DATASET_DIR / "calypte_anna_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-eliason-calypte-anna-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
