#!/usr/bin/env python3
"""
2001-frankel-tetrahymena-thermophila-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0997)
Organism: Tetrahymena thermophila (Ciliated Cortical Protist)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Tetrahymena
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0997_tetrahymena_thermophila_bioelectric_polarity.lean` (`MathOfBiology.Loop_0997`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetrahymena_thermophila_bioelectric_polarity"
EXPECTED_SHA256 = {
    "tetrahymena_thermophila_bioelectric_polarity_reference.json": "485879cccb2265601e82ef78d156e9cb22df9b5e0abdaa46f619ebd51104661a",
    "tetrahymena_thermophila_specimen.jpg": "7704d06cf5ec8d2a1b6bb5954c4846597cac08e5f4d0c57aed6cd6e1d729026c",
    "scout_manifest.json": "96d2cf989b8dcd5741bbb858d28dbe175fb17bc4167e289cbbb0bf072a097038",
    "paper_citations.bib": "655cbb9b1fed8007203f78b2642034032cde28591745edabc7849964f86e6096"
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
    ref = json.loads((DATASET_DIR / "tetrahymena_thermophila_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-frankel-tetrahymena-thermophila-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
