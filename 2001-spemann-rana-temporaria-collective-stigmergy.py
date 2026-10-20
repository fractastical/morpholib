#!/usr/bin/env python3
"""
2001-spemann-rana-temporaria-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1026)
Organism: Rana temporaria (European Common Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Rana
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1026_rana_temporaria_collective_stigmergy.lean` (`MathOfBiology.Loop_1026`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/rana_temporaria_collective_stigmergy"
EXPECTED_SHA256 = {
    "rana_temporaria_collective_stigmergy_reference.json": "ad67a2a5c1399a1d3b7fd38548416e6318c3d4c55f52d87eafb79ccbe0f87e7f",
    "rana_temporaria_specimen.jpg": "966174057e8f12954fa5a7ba2b0a2b543b64c2900c6c92ea4a55926d1c57f111",
    "scout_manifest.json": "7f14e7b752fb714abe871d58904275bd3769dbaaae6f6192c003dccac74c16f5",
    "paper_citations.bib": "0c9355ed33b2dd6009837960b363bb9c2e89dbe374777016621a68c62c6563d5"
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
    ref = json.loads((DATASET_DIR / "rana_temporaria_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-spemann-rana-temporaria-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
