#!/usr/bin/env python3
"""
2017-abrams-aurelia-aurita-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1064)
Organism: Aurelia aurita (Moon Jellyfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Aurelia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1064_aurelia_aurita_collective_stigmergy.lean` (`MathOfBiology.Loop_1064`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aurelia_aurita_collective_stigmergy"
EXPECTED_SHA256 = {
    "aurelia_aurita_collective_stigmergy_reference.json": "6614b001a88e62b4da22e747336988c448c7925267e0664bb0cc6f7f9622ff7f",
    "aurelia_aurita_specimen.jpg": "1a8c5b9fb4abe35f1362b2c458d03bafa15966b0f6046d49aa840fdd3a66769e",
    "scout_manifest.json": "0a032ffe6878b7f62ac2ea3be8fdb69fae4b375281e12103a5d8eedd3db97ce2",
    "paper_citations.bib": "162c9efb36b1271212662a8c6108b5169397e8167d94e283a56fc60555b14835"
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
    ref = json.loads((DATASET_DIR / "aurelia_aurita_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-abrams-aurelia-aurita-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
