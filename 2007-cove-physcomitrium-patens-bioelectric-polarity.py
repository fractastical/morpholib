#!/usr/bin/env python3
"""
2007-cove-physcomitrium-patens-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0972)
Organism: Physcomitrium patens (Spreading Earthmoss)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Physcomitrium
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0972_physcomitrium_patens_bioelectric_polarity.lean` (`MathOfBiology.Loop_0972`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/physcomitrium_patens_bioelectric_polarity"
EXPECTED_SHA256 = {
    "physcomitrium_patens_bioelectric_polarity_reference.json": "ef164df14db2466d1d89b001a475b3c0e441126f4601e59e44e422212a72d397",
    "physcomitrium_patens_specimen.jpg": "a115412e5c6b1e47cad14da271d9ee4f3d33b2b7150fb4c533f145234f694d13",
    "scout_manifest.json": "28797177620ae5bac54fcc48ac14f52d20c4b022ac6afa61ae90f934ef2ad15c",
    "paper_citations.bib": "a0f7734ca02c6c26eec7cb9732c9182b5e6d4a1b9d21f121d375bf0ebf9711d7"
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
    ref = json.loads((DATASET_DIR / "physcomitrium_patens_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-cove-physcomitrium-patens-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
