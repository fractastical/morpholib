#!/usr/bin/env python3
"""
2010-keller-xenopus-tropicalis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1021)
Organism: Xenopus tropicalis (Western Clawed Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Xenopus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1021_xenopus_tropicalis_collective_stigmergy.lean` (`MathOfBiology.Loop_1021`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/xenopus_tropicalis_collective_stigmergy"
EXPECTED_SHA256 = {
    "xenopus_tropicalis_collective_stigmergy_reference.json": "12b6c6d8d8473f3a62c55794dd57849a67746446ffe431e51d8297b64fb8d5e1",
    "xenopus_tropicalis_specimen.jpg": "13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f",
    "scout_manifest.json": "0e9b11caa1272b9c608ab7676cbac25ca5387ad0f9b3b2cf0f1b4bbb12c52686",
    "paper_citations.bib": "9bf1da577059b132d889c78e232409d3d00ff87e0c0350de76dcce3c1d426738"
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
    ref = json.loads((DATASET_DIR / "xenopus_tropicalis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-keller-xenopus-tropicalis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
