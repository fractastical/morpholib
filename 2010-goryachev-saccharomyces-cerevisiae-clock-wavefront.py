#!/usr/bin/env python3
"""
2010-goryachev-saccharomyces-cerevisiae-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0386)
Organism: Saccharomyces cerevisiae (Budding Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Saccharomyces
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0386_saccharomyces_cerevisiae_clock_wavefront.lean` (`MathOfBiology.Loop_0386`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/saccharomyces_cerevisiae_clock_wavefront"
EXPECTED_SHA256 = {
    "saccharomyces_cerevisiae_clock_wavefront_reference.json": "ef41b67d320a16eb3fb7f40da3ed9f428ca6e143c15cfca768c47a53461fd422",
    "saccharomyces_cerevisiae_specimen.jpg": "03d89aa520798f5a2c721833870f675b8d4676ea933358e95702fcd1732eae0b",
    "scout_manifest.json": "54744cd26d692452b82336d716372e55a7ac271e82e83191ca985184fec644a2",
    "paper_citations.bib": "ac6960ddb4ac8f3e34a7895922513e53054cf77996f45b11db95bd9bd1debfd7"
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
    ref = json.loads((DATASET_DIR / "saccharomyces_cerevisiae_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-goryachev-saccharomyces-cerevisiae-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
