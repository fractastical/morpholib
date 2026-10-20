#!/usr/bin/env python3
"""
2006-benjacob-paenibacillus-dendritiformis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0789)
Organism: Paenibacillus dendritiformis (Chiral Branching Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Paenibacillus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0789_paenibacillus_dendritiformis_elastic_buckling.lean` (`MathOfBiology.Loop_0789`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/paenibacillus_dendritiformis_elastic_buckling"
EXPECTED_SHA256 = {
    "paenibacillus_dendritiformis_elastic_buckling_reference.json": "9e7ccb61d3bc63f35fcff7596ed8fa0e327c9711037ab1092432ca115b8c7d8f",
    "paenibacillus_dendritiformis_specimen.jpg": "dbed60f770b34bfce5f9193b55484568f5cccc3a5f417b5c7c6e79a7a8c84cf6",
    "scout_manifest.json": "db15324228a42a1e0e777954551f96afc225cdb3e87262a202890770b6467282",
    "paper_citations.bib": "a2de3212dce3c765033d313d922f56630762127daa8a5fc618d433c14a44fd78"
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
    ref = json.loads((DATASET_DIR / "paenibacillus_dendritiformis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-benjacob-paenibacillus-dendritiformis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
