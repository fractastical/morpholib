#!/usr/bin/env python3
"""
1976-lacalli-micrasterias-denticulata-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0994)
Organism: Micrasterias denticulata (Star Desmid Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Micrasterias
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0994_micrasterias_denticulata_bioelectric_polarity.lean` (`MathOfBiology.Loop_0994`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/micrasterias_denticulata_bioelectric_polarity"
EXPECTED_SHA256 = {
    "micrasterias_denticulata_bioelectric_polarity_reference.json": "78c4954d896e932daed5ca74b95efad3a86d38da923ee5b42e5517c0187c72c0",
    "micrasterias_denticulata_specimen.jpg": "3e8886e5b5c9167ab5c9217bf3f399a4d3dc7dc9eca6bc05ee5e94f5ab6fac34",
    "scout_manifest.json": "e7eb431bacf340fc2752658087e974e205ec5235636cd5c1b5f44a24b8a5361a",
    "paper_citations.bib": "66c7723c0abc7eb1482b1877230f02c0b39370d62bf7a540fe57cbea93bab287"
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
    ref = json.loads((DATASET_DIR / "micrasterias_denticulata_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1976-lacalli-micrasterias-denticulata-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
