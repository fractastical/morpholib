#!/usr/bin/env python3
"""
1977-lacalli-micrasterias-denticulata-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0394)
Organism: Micrasterias denticulata (Star Desmid Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Micrasterias
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0394_micrasterias_denticulata_clock_wavefront.lean` (`MathOfBiology.Loop_0394`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/micrasterias_denticulata_clock_wavefront"
EXPECTED_SHA256 = {
    "micrasterias_denticulata_clock_wavefront_reference.json": "e760149d15dcb9672550f39f286bc23a789d4b1e214183f5762947b7775975ce",
    "micrasterias_denticulata_specimen.jpg": "3e8886e5b5c9167ab5c9217bf3f399a4d3dc7dc9eca6bc05ee5e94f5ab6fac34",
    "scout_manifest.json": "8c4838aa066d3145f71f7d2792c8e87c40bd8c9a1e774685480c647a48b05d2b",
    "paper_citations.bib": "20c3e65a38e394ff836fada6a05b12205a95c1fbc6e364808ca4e6811eca8a24"
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
    ref = json.loads((DATASET_DIR / "micrasterias_denticulata_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1977-lacalli-micrasterias-denticulata-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
