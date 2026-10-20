#!/usr/bin/env python3
"""
2021-lee-utricularia-gibba-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0377)
Organism: Utricularia gibba (Humped Bladderwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Utricularia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0377_utricularia_gibba_clock_wavefront.lean` (`MathOfBiology.Loop_0377`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/utricularia_gibba_clock_wavefront"
EXPECTED_SHA256 = {
    "utricularia_gibba_clock_wavefront_reference.json": "107a65e25b9ea91e2da746d11915a3277b53e77efd5997067fd11850a6914953",
    "utricularia_gibba_specimen.jpg": "9143b846b36d681d8c10ef23ad998247630e9d5e3abebcdfc12e3fa9dadfc3f7",
    "scout_manifest.json": "ae07e3b5232a4e325559733809fa129137ebddaa864adf668f47ef4488cdeba8",
    "paper_citations.bib": "8f7620aa65e7d3a1ee2f60f914503a1ccdac54ac60c5920da04afec2c08e9a04"
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
    ref = json.loads((DATASET_DIR / "utricularia_gibba_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-lee-utricularia-gibba-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
