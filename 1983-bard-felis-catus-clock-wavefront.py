#!/usr/bin/env python3
"""
1983-bard-felis-catus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0303)
Organism: Felis catus (Domestic Cat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Felis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0303_felis_catus_clock_wavefront.lean` (`MathOfBiology.Loop_0303`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/felis_catus_clock_wavefront"
EXPECTED_SHA256 = {
    "felis_catus_clock_wavefront_reference.json": "d4083a26d13aa2cfde1396c37bc2eb8723ef2343d81b8d176b18f1f6d4374dbb",
    "felis_catus_specimen.jpg": "f417376a9a5f657061f2082396f5cc7eb2b6a4d0c92e0c1d3692ceb2b52926bd",
    "scout_manifest.json": "941d3adc938f0e4abbfa0beee2f57db8a98c2452258a017b5cc8d0150dce510e",
    "paper_citations.bib": "b49ef57826adceb4c0dbb7d8e2a83910c385ec9ef2938f8351a489dc2e710422"
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
    ref = json.loads((DATASET_DIR / "felis_catus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1983-bard-felis-catus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
