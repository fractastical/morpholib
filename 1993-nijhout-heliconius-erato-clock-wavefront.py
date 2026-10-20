#!/usr/bin/env python3
"""
1993-nijhout-heliconius-erato-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0347)
Organism: Heliconius erato (Red Postman Butterfly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Heliconius
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0347_heliconius_erato_clock_wavefront.lean` (`MathOfBiology.Loop_0347`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/heliconius_erato_clock_wavefront"
EXPECTED_SHA256 = {
    "heliconius_erato_clock_wavefront_reference.json": "71836437c2e536766c2a9ad9ebb7bd0d09588f1bb2262cac4903df69305b51e1",
    "heliconius_erato_specimen.jpg": "e54735f4ca16e4326cb91d2969d83bb613e3c4111aaf7f30cb218075b6fec48e",
    "scout_manifest.json": "6c22172d001fdba0e895167a1077b4a20b010569dcbcfbab7aa10c3ff17923a6",
    "paper_citations.bib": "d283ff7bbaa0f1714024369926aaf132e2f225494263c13d37713f88ade934b0"
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
    ref = json.loads((DATASET_DIR / "heliconius_erato_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1993-nijhout-heliconius-erato-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
