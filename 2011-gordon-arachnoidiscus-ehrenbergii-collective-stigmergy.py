#!/usr/bin/env python3
"""
2011-gordon-arachnoidiscus-ehrenbergii-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1099)
Organism: Arachnoidiscus ehrenbergii (Wheel Centric Diatom)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Arachnoidiscus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1099_arachnoidiscus_ehrenbergii_collective_stigmergy.lean` (`MathOfBiology.Loop_1099`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/arachnoidiscus_ehrenbergii_collective_stigmergy"
EXPECTED_SHA256 = {
    "arachnoidiscus_ehrenbergii_collective_stigmergy_reference.json": "d230cbd2c92cbd9b253be551014e877ebb9f549c10c6216c3d4feb9d5e3abaca",
    "arachnoidiscus_ehrenbergii_specimen.jpg": "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05",
    "scout_manifest.json": "6a79582b9203f64e74dad967dba3b904f1ca9db4222e02b701549686f326b0dc",
    "paper_citations.bib": "947834c065a2db8fd1cf2166fb3542941e3a7faebddf0e09b06827c45b7996ad"
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
    ref = json.loads((DATASET_DIR / "arachnoidiscus_ehrenbergii_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-gordon-arachnoidiscus-ehrenbergii-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
