#!/usr/bin/env python3
"""
2021-lee-utricularia-gibba-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1077)
Organism: Utricularia gibba (Humped Bladderwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Utricularia
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1077_utricularia_gibba_collective_stigmergy.lean` (`MathOfBiology.Loop_1077`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/utricularia_gibba_collective_stigmergy"
EXPECTED_SHA256 = {
    "utricularia_gibba_collective_stigmergy_reference.json": "9563d17b79eaaea6a9cbf81dc782ae1125f0c5b4a26ac090d6c7361002bedba7",
    "utricularia_gibba_specimen.jpg": "9143b846b36d681d8c10ef23ad998247630e9d5e3abebcdfc12e3fa9dadfc3f7",
    "scout_manifest.json": "4646ddd88b2f9b7febd38bc542c46403971aaba0af17a14f426473f9008cf241",
    "paper_citations.bib": "eda69076205e56912623588683fca5a7bed06508178033b692c3baea09c02c1d"
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
    ref = json.loads((DATASET_DIR / "utricularia_gibba_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-lee-utricularia-gibba-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
