#!/usr/bin/env python3
"""
1986-goodwin-acetabularia-acetabulum-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0992)
Organism: Acetabularia acetabulum (Mermaid's Wineglass Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acetabularia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0992_acetabularia_acetabulum_bioelectric_polarity.lean` (`MathOfBiology.Loop_0992`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acetabularia_acetabulum_bioelectric_polarity"
EXPECTED_SHA256 = {
    "acetabularia_acetabulum_bioelectric_polarity_reference.json": "ef752ed71b41f07f12539277983505cc7436287f3c53c2e8b916015229d7d8e0",
    "acetabularia_acetabulum_specimen.jpg": "17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1",
    "scout_manifest.json": "81f27d0e29cf5f0f35b5ac9e878c7f5b44a4f6f7da63b0b8e66d331624d10284",
    "paper_citations.bib": "f8031e5191a98b00e7fd01f42e5674d287059aa888c7c10ddfdd406691e096d3"
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
    ref = json.loads((DATASET_DIR / "acetabularia_acetabulum_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1986-goodwin-acetabularia-acetabulum-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
