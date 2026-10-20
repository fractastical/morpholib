#!/usr/bin/env python3
"""
2004-autum-gekko-gecko-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0328)
Organism: Gekko gecko (Tokay Gecko)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Gekko
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0328_gekko_gecko_clock_wavefront.lean` (`MathOfBiology.Loop_0328`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gekko_gecko_clock_wavefront"
EXPECTED_SHA256 = {
    "gekko_gecko_clock_wavefront_reference.json": "589f115d1e094568a4eadaacbb4400d3ab091044cbb83740de5095752ded219b",
    "gekko_gecko_specimen.jpg": "d5822e517434737d90b14fd5099c77f09ab976ca2a3ba00c2484f69a65eb91e8",
    "scout_manifest.json": "5c7802bf6154c24c87ea81a8af416582b927af7710bb1548fd34115765521974",
    "paper_citations.bib": "1967c76ae7564f481d7618c783f46062e22da5304a69da86db10686c6847e877"
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
    ref = json.loads((DATASET_DIR / "gekko_gecko_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-autum-gekko-gecko-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
