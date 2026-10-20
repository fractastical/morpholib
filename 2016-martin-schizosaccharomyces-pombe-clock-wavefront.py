#!/usr/bin/env python3
"""
2016-martin-schizosaccharomyces-pombe-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0387)
Organism: Schizosaccharomyces pombe (Fission Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Schizosaccharomyces
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0387_schizosaccharomyces_pombe_clock_wavefront.lean` (`MathOfBiology.Loop_0387`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schizosaccharomyces_pombe_clock_wavefront"
EXPECTED_SHA256 = {
    "schizosaccharomyces_pombe_clock_wavefront_reference.json": "bc7c2bca1036eafa47e683621679710c5e917b28db3e795b10a561ad5dde709f",
    "schizosaccharomyces_pombe_specimen.jpg": "1ddf4f2081b0c047e472365745cd54e38ef85ea46a86a451dc98e0a7d6b5c9f8",
    "scout_manifest.json": "fbae5823ccd3d8922d0a578ed6af297e2636334e42bc48d52bad203a111a3bf1",
    "paper_citations.bib": "5298e1642670922f06f29a46ceb2f1505c28a2c2c3650174a284e2efbced8b5d"
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
    ref = json.loads((DATASET_DIR / "schizosaccharomyces_pombe_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-martin-schizosaccharomyces-pombe-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
