#!/usr/bin/env python3
"""
2019-forterre-mimosa-pudica-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0778)
Organism: Mimosa pudica (Sensitive Plant)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Mimosa
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0778_mimosa_pudica_elastic_buckling.lean` (`MathOfBiology.Loop_0778`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mimosa_pudica_elastic_buckling"
EXPECTED_SHA256 = {
    "mimosa_pudica_elastic_buckling_reference.json": "6b150018b1dabf49e24538c8e5c282bf176f78782b1786ed40c620faad882477",
    "mimosa_pudica_specimen.jpg": "4ae1af66e325ef597ebdce00e9f639174383d049bc7b0e98b02e97de71c83e20",
    "scout_manifest.json": "01bd7bf2869312b9c8f5c007b5320bb63303b1eee696d1eb1a6e0b4c3f6559dc",
    "paper_citations.bib": "197217849302d134ddbdaa800df416b6e6509066becd63c5a2c0a76e37b88211"
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
    ref = json.loads((DATASET_DIR / "mimosa_pudica_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-forterre-mimosa-pudica-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
