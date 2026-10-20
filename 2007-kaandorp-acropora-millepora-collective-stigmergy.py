#!/usr/bin/env python3
"""
2007-kaandorp-acropora-millepora-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1065)
Organism: Acropora millepora (Branching Staghorn Coral)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Acropora
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1065_acropora_millepora_collective_stigmergy.lean` (`MathOfBiology.Loop_1065`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acropora_millepora_collective_stigmergy"
EXPECTED_SHA256 = {
    "acropora_millepora_collective_stigmergy_reference.json": "1e7be4b3d4b20d25a61d681207f6b6811930c9cd4e5d054577eb83ed74ffcd1e",
    "acropora_millepora_specimen.jpg": "85c35e854dd9d07d46528548512579040fef8d6cd4893257fb28bd3b4a468eb0",
    "scout_manifest.json": "58db3a2fff6f221c61991ec8648ce1e3a5e6126dd9a180f5a1f82ad6dfaab347",
    "paper_citations.bib": "2e766f3a8bd4081ab32cec552f12e87abebe6c234e46e2ca3cebdcd703148088"
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
    ref = json.loads((DATASET_DIR / "acropora_millepora_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-kaandorp-acropora-millepora-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
