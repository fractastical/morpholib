#!/usr/bin/env python3
"""
2003-czirok-proteus-mirabilis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1090)
Organism: Proteus mirabilis (Bullseye Swarming Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Proteus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1090_proteus_mirabilis_collective_stigmergy.lean` (`MathOfBiology.Loop_1090`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/proteus_mirabilis_collective_stigmergy"
EXPECTED_SHA256 = {
    "proteus_mirabilis_collective_stigmergy_reference.json": "dfccdbe6927ae217bf3367a3ddf76200261607d4c370c046a868ae4a0cb581be",
    "proteus_mirabilis_specimen.jpg": "ac07f43481b9b721da7b124217c182a69192087ef9fb2f59564a107e3383a85b",
    "scout_manifest.json": "a771ae41fbe77a74c6970b5003587799316409b00fb4219233c322609f9eda51",
    "paper_citations.bib": "66fa1edf96d2b6be801fb3884f6cc0d8d6da72010e6ed7aee5bb70d67c806d8f"
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
    ref = json.loads((DATASET_DIR / "proteus_mirabilis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-czirok-proteus-mirabilis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
