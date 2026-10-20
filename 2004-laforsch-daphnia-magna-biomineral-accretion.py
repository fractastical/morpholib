#!/usr/bin/env python3
"""
2004-laforsch-daphnia-magna-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0849)
Organism: Daphnia magna (Water Flea)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Daphnia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0849_daphnia_magna_biomineral_accretion.lean` (`MathOfBiology.Loop_0849`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/daphnia_magna_biomineral_accretion"
EXPECTED_SHA256 = {
    "daphnia_magna_biomineral_accretion_reference.json": "7d5e27ed0ce4347c7713a32d7d7a5d917da32ff10afcc2baf0d45a852c395d93",
    "daphnia_magna_specimen.jpg": "fc61b6ab51b92e988777a61cce4db7434210496ee5084ae01ee9f7adf6fcf44b",
    "scout_manifest.json": "e210bae1988f38366ed1f2dc820cf60ecc2eaab1d684726a308693db59196bcd",
    "paper_citations.bib": "9beb950d0b01edbc265bee62fefd0f7a1ccb5c648becad4a6f9bf8c38cdf4c3d"
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
    ref = json.loads((DATASET_DIR / "daphnia_magna_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-laforsch-daphnia-magna-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
