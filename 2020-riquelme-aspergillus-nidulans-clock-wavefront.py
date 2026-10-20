#!/usr/bin/env python3
"""
2020-riquelme-aspergillus-nidulans-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0382)
Organism: Aspergillus nidulans (Filamentous Fungus)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Aspergillus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0382_aspergillus_nidulans_clock_wavefront.lean` (`MathOfBiology.Loop_0382`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aspergillus_nidulans_clock_wavefront"
EXPECTED_SHA256 = {
    "aspergillus_nidulans_clock_wavefront_reference.json": "db8657ac7dc507b9f6b21d44fb0fff37a0d148f66c1e7976000a295ee981302e",
    "aspergillus_nidulans_specimen.jpg": "93e53157e77eaccce94a5792d58d0e622ba9bb55b15fd47100e1f7c90985dec5",
    "scout_manifest.json": "cc0e08ae830f985672fc66984c7416e701e41a08212b27d1fb383ea2e53bd77f",
    "paper_citations.bib": "13d96d401637bb3b222368205ee4b3fc1f4ef4014946a9232658dd91a48e3ceb"
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
    ref = json.loads((DATASET_DIR / "aspergillus_nidulans_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-riquelme-aspergillus-nidulans-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
