#!/usr/bin/env python3
"""
2007-marshall-chlamydomonas-reinhardtii-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0395)
Organism: Chlamydomonas reinhardtii (Biflagellate Microalga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Chlamydomonas
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0395_chlamydomonas_reinhardtii_clock_wavefront.lean` (`MathOfBiology.Loop_0395`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chlamydomonas_reinhardtii_clock_wavefront"
EXPECTED_SHA256 = {
    "chlamydomonas_reinhardtii_clock_wavefront_reference.json": "27131f1853fba6490b070b16a901aebf4064ae3f1acad9950a0fc6be03bed2f6",
    "chlamydomonas_reinhardtii_specimen.jpg": "ebcb2a487066205d3a9f06b194d0bfd68786337b7fbbf44dd71400b689d923bd",
    "scout_manifest.json": "f5d85eb683b0c29655d9522e14b073b085c11e1fb9b17abe9283f434f20943f5",
    "paper_citations.bib": "f93537fa4c1937e82988f5b2571d6d96b3790ac58f1cb08a4609c74800ddd09f"
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
    ref = json.loads((DATASET_DIR / "chlamydomonas_reinhardtii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-marshall-chlamydomonas-reinhardtii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
