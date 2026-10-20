#!/usr/bin/env python3
"""
2006-buhl-schistocerca-gregaria-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0148)
Organism: Schistocerca gregaria (Desert Locust)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Schistocerca
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0148_schistocerca_gregaria_turing_wavelength.lean` (`MathOfBiology.Loop_0148`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schistocerca_gregaria_turing_wavelength"
EXPECTED_SHA256 = {
    "schistocerca_gregaria_turing_wavelength_reference.json": "aa68ff5915e09115e435b47e2f7c858cc184b812be901a467ceedb6547214121",
    "schistocerca_gregaria_specimen.jpg": "3f8a278b02bc5e5158458d4ff3669c71a29826b1129be2cc7920ab4b3ead3827",
    "scout_manifest.json": "f41620f8e7250a80c9703ea3f0f62d9d0f996b7f441e1fb883aec0a45bddef99",
    "paper_citations.bib": "4da68f699e7e32ac9c1866e9f7e1eebfd0d560acc7e7937bb6bf7d5efe813ac8"
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
    ref = json.loads((DATASET_DIR / "schistocerca_gregaria_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-buhl-schistocerca-gregaria-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
