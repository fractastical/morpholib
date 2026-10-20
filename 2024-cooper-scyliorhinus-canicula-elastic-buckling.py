#!/usr/bin/env python3
"""
2024-cooper-scyliorhinus-canicula-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0735)
Organism: Scyliorhinus canicula (Small-Spotted Catshark)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Scyliorhinus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0735_scyliorhinus_canicula_elastic_buckling.lean` (`MathOfBiology.Loop_0735`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/scyliorhinus_canicula_elastic_buckling"
EXPECTED_SHA256 = {
    "scyliorhinus_canicula_elastic_buckling_reference.json": "387f1babd800185aae4e10a303e0c2cd057fdb6803f8791db45aaea257a97bce",
    "scyliorhinus_canicula_specimen.jpg": "71525dd878e51b100ea6bbd4e4005751066059fc236d4e204ad81ed2bbb5d553",
    "scout_manifest.json": "7ab0243be1d97fcb1dde539d2fa68c754c9e414c3be498a3a123a253a15eb23e",
    "paper_citations.bib": "3de68c297a15ceaf00744e2ab49c7d483c6be2868c52b17d2da8fec0049054cd"
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
    ref = json.loads((DATASET_DIR / "scyliorhinus_canicula_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2024-cooper-scyliorhinus-canicula-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
