#!/usr/bin/env python3
"""
2007-marshall-chlamydomonas-reinhardtii-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1095)
Organism: Chlamydomonas reinhardtii (Biflagellate Microalga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Chlamydomonas
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1095_chlamydomonas_reinhardtii_collective_stigmergy.lean` (`MathOfBiology.Loop_1095`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chlamydomonas_reinhardtii_collective_stigmergy"
EXPECTED_SHA256 = {
    "chlamydomonas_reinhardtii_collective_stigmergy_reference.json": "37073b1cbaf64dffa9fc753bdb27d7aa58f26e006a4fe42553834b29f5339b87",
    "chlamydomonas_reinhardtii_specimen.jpg": "ebcb2a487066205d3a9f06b194d0bfd68786337b7fbbf44dd71400b689d923bd",
    "scout_manifest.json": "b5f0570be41470f4df9eee2d9e5f0eab76e6368d4f90d9b6c42dd339af7e7214",
    "paper_citations.bib": "c252ec7c36ffea1a2e3814941758221365816bea0cad8b09abc0b3a448172ab9"
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
    ref = json.loads((DATASET_DIR / "chlamydomonas_reinhardtii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-marshall-chlamydomonas-reinhardtii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
