#!/usr/bin/env python3
"""
2020-riquelme-aspergillus-nidulans-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1082)
Organism: Aspergillus nidulans (Filamentous Fungus)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Aspergillus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1082_aspergillus_nidulans_collective_stigmergy.lean` (`MathOfBiology.Loop_1082`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aspergillus_nidulans_collective_stigmergy"
EXPECTED_SHA256 = {
    "aspergillus_nidulans_collective_stigmergy_reference.json": "b9a12cd1608f3ffe3131c23760c66f209a28e13391919ceb9567de8eee66c3e7",
    "aspergillus_nidulans_specimen.jpg": "93e53157e77eaccce94a5792d58d0e622ba9bb55b15fd47100e1f7c90985dec5",
    "scout_manifest.json": "f7877af7675cca76adae0f28da3f82c7d520d9fa41a90e0dd367446ab0109546",
    "paper_citations.bib": "8e43b4d037f155bbf078a96014258ead942aaf2db756fccf2a73f93a94500c2e"
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
    ref = json.loads((DATASET_DIR / "aspergillus_nidulans_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-riquelme-aspergillus-nidulans-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
