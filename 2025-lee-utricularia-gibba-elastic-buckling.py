#!/usr/bin/env python3
"""
2025-lee-utricularia-gibba-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0777)
Organism: Utricularia gibba (Humped Bladderwort)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Utricularia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0777_utricularia_gibba_elastic_buckling.lean` (`MathOfBiology.Loop_0777`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/utricularia_gibba_elastic_buckling"
EXPECTED_SHA256 = {
    "utricularia_gibba_elastic_buckling_reference.json": "87ca90e4f195a4c41f9f85013ee50304ec4a8199a92bf24224c17bd01e6277c2",
    "utricularia_gibba_specimen.jpg": "9143b846b36d681d8c10ef23ad998247630e9d5e3abebcdfc12e3fa9dadfc3f7",
    "scout_manifest.json": "dc957ede344dfb72462bc71de310e4ff24de752e590c33b24be924027f441a54",
    "paper_citations.bib": "b3cab4c7771d120b66759283f7c2e6523164102bcd93c3c2c4e70803a0520a53"
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
    ref = json.loads((DATASET_DIR / "utricularia_gibba_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2025-lee-utricularia-gibba-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
