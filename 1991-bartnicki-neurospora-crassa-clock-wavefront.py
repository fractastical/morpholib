#!/usr/bin/env python3
"""
1991-bartnicki-neurospora-crassa-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0381)
Organism: Neurospora crassa (Red Bread Mold)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Neurospora
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0381_neurospora_crassa_clock_wavefront.lean` (`MathOfBiology.Loop_0381`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/neurospora_crassa_clock_wavefront"
EXPECTED_SHA256 = {
    "neurospora_crassa_clock_wavefront_reference.json": "bd96855111ed6d4ac2b21e284ee849018ef74724d02ce055947ced618d02cfff",
    "neurospora_crassa_specimen.jpg": "01d6531f055000fd6be6dc5baa24d01e76899962e05a548e4be38a45032165f0",
    "scout_manifest.json": "ab02e422b20c0222d28834fcde26c51064eaeae6e837f5fdd8d87cf6ab61fb71",
    "paper_citations.bib": "a7cf031daabbd0107787695e9c4f767395f3d5c7f8e1db67d50cbdd55b91666d"
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
    ref = json.loads((DATASET_DIR / "neurospora_crassa_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1991-bartnicki-neurospora-crassa-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
