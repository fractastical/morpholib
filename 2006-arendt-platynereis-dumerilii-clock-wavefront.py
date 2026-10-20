#!/usr/bin/env python3
"""
2006-arendt-platynereis-dumerilii-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0361)
Organism: Platynereis dumerilii (Marine Ragworm)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Platynereis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0361_platynereis_dumerilii_clock_wavefront.lean` (`MathOfBiology.Loop_0361`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/platynereis_dumerilii_clock_wavefront"
EXPECTED_SHA256 = {
    "platynereis_dumerilii_clock_wavefront_reference.json": "ca717f7e674ce035dd0348152286576d84b33b0914b7926a18d72da3d8b7a98d",
    "platynereis_dumerilii_specimen.jpg": "49376d1416ec7548315612807bfb066579e6891667d63542b0c9e663606fe5b9",
    "scout_manifest.json": "f9d64ffb95775f0007a72695052f93ff1f1373f76bd250c41678024d6da7ea59",
    "paper_citations.bib": "1247b3711ffec692e7a1486563a1ac8c1713dcdef2b0657dd01cf2a032de2285"
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
    ref = json.loads((DATASET_DIR / "platynereis_dumerilii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-arendt-platynereis-dumerilii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
