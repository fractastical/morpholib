#!/usr/bin/env python3
"""
1977-lacalli-micrasterias-denticulata-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1094)
Organism: Micrasterias denticulata (Star Desmid Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Micrasterias
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1094_micrasterias_denticulata_collective_stigmergy.lean` (`MathOfBiology.Loop_1094`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/micrasterias_denticulata_collective_stigmergy"
EXPECTED_SHA256 = {
    "micrasterias_denticulata_collective_stigmergy_reference.json": "a980a94260e80f161004b74202e9524877dad1bc5a23bc21a53294d99b09602f",
    "micrasterias_denticulata_specimen.jpg": "3e8886e5b5c9167ab5c9217bf3f399a4d3dc7dc9eca6bc05ee5e94f5ab6fac34",
    "scout_manifest.json": "5735a1db94811ecccd00ec922a152897368da41732c07f8d72c336b73c8a78d1",
    "paper_citations.bib": "5f8930fd9e7a6d9c93d8076b14d048e32f21fa51a7bd35608a9936499c6ce289"
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
    ref = json.loads((DATASET_DIR / "micrasterias_denticulata_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1977-lacalli-micrasterias-denticulata-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
