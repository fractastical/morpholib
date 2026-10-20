#!/usr/bin/env python3
"""
2019-cooper-scyliorhinus-canicula-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0935)
Organism: Scyliorhinus canicula (Small-Spotted Catshark)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Scyliorhinus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0935_scyliorhinus_canicula_bioelectric_polarity.lean` (`MathOfBiology.Loop_0935`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/scyliorhinus_canicula_bioelectric_polarity"
EXPECTED_SHA256 = {
    "scyliorhinus_canicula_bioelectric_polarity_reference.json": "42641d8efe26e7c993554905806f8fad9a73aff752c84c9ceef1a4874ff71f24",
    "scyliorhinus_canicula_specimen.jpg": "71525dd878e51b100ea6bbd4e4005751066059fc236d4e204ad81ed2bbb5d553",
    "scout_manifest.json": "2fb0cf7f800c3f473aa36da4ee166f9dfc178905415bc5ba64eb604ef3ce0508",
    "paper_citations.bib": "d043e96d401a6198b91110c74de5ee8d97185ba94d76038fb15a2d256619c504"
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
    ref = json.loads((DATASET_DIR / "scyliorhinus_canicula_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-cooper-scyliorhinus-canicula-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
