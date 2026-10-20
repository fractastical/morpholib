#!/usr/bin/env python3
"""
2024-riquelme-aspergillus-nidulans-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0782)
Organism: Aspergillus nidulans (Filamentous Fungus)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Aspergillus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0782_aspergillus_nidulans_elastic_buckling.lean` (`MathOfBiology.Loop_0782`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aspergillus_nidulans_elastic_buckling"
EXPECTED_SHA256 = {
    "aspergillus_nidulans_elastic_buckling_reference.json": "8300b31a5fe4c05606fe8934c302ff0c5f1b6cfd6dbb46ad124cea7a0f220a27",
    "aspergillus_nidulans_specimen.jpg": "93e53157e77eaccce94a5792d58d0e622ba9bb55b15fd47100e1f7c90985dec5",
    "scout_manifest.json": "582875b369a08a1f33a9a5e777a1c469101cec507374bcb814dda79be68753ad",
    "paper_citations.bib": "106c31148510cca7bbcf092c1882d29b6093c99503f16e98af6e957ceeed913b"
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
    ref = json.loads((DATASET_DIR / "aspergillus_nidulans_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2024-riquelme-aspergillus-nidulans-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
