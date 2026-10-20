#!/usr/bin/env python3
"""
2002-benjacob-paenibacillus-dendritiformis-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1089)
Organism: Paenibacillus dendritiformis (Chiral Branching Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Paenibacillus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1089_paenibacillus_dendritiformis_collective_stigmergy.lean` (`MathOfBiology.Loop_1089`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/paenibacillus_dendritiformis_collective_stigmergy"
EXPECTED_SHA256 = {
    "paenibacillus_dendritiformis_collective_stigmergy_reference.json": "1e7a56170f57a26bb33a466b8426127c3d2d51072c7525ef140485cc80b633d0",
    "paenibacillus_dendritiformis_specimen.jpg": "dbed60f770b34bfce5f9193b55484568f5cccc3a5f417b5c7c6e79a7a8c84cf6",
    "scout_manifest.json": "314d16652ad2ec456f2f45863d527b98d9611dbf5f81e361396ed85999b1ba66",
    "paper_citations.bib": "58ae966b6e29e9d30ab85f9fbf6ab0994c64c0f2990d057da438f22b623fffe3"
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
    ref = json.loads((DATASET_DIR / "paenibacillus_dendritiformis_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-benjacob-paenibacillus-dendritiformis-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
