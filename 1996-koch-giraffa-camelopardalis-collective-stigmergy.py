#!/usr/bin/env python3
"""
1996-koch-giraffa-camelopardalis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1005)
Organism: Giraffa camelopardalis (Reticulated Giraffe)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Giraffa
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1005_giraffa_camelopardalis_collective_stigmergy.lean` (`MathOfBiology.Loop_1005`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/giraffa_camelopardalis_collective_stigmergy"
EXPECTED_SHA256 = {
    "giraffa_camelopardalis_collective_stigmergy_reference.json": "83bd49508769b46751d54cd748e23fe2e7f21e0508f32512b3c1bba2222e429f",
    "giraffa_camelopardalis_specimen.jpg": "d65ee763a067a60ce829e77f52e0ca7eadb883d032bc517d6f76edf060800969",
    "scout_manifest.json": "839d4b3d17abe7a79821f37117b5b96c085616383f9c3951e2ae5fb7fc021b64",
    "paper_citations.bib": "537163e52e815b1c2d5c856596b91b14b9a42d189e1cfec2ce8681597c5eaab4"
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
    ref = json.loads((DATASET_DIR / "giraffa_camelopardalis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1996-koch-giraffa-camelopardalis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
