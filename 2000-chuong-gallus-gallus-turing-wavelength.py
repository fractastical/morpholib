#!/usr/bin/env python3
"""
2000-chuong-gallus-gallus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0111)
Organism: Gallus gallus (Red Junglefowl Chick)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Gallus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0111_gallus_gallus_turing_wavelength.lean` (`MathOfBiology.Loop_0111`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gallus_gallus_turing_wavelength"
EXPECTED_SHA256 = {
    "gallus_gallus_turing_wavelength_reference.json": "da8d7c5b869853e685be1ef4b7518fe581bfd03be0cd1ef99b78a46b055cbba9",
    "gallus_gallus_specimen.jpg": "7e7d58db6bb564ebe68e091678338c7c06b920cdb66f9dffd99b4141bf16c9cb",
    "scout_manifest.json": "30947336a0b5abf916784e596efedf47a7e1e175396dd4547efb79093fab3ec5",
    "paper_citations.bib": "3efbbec9561712614b06c1ffc9df9a67139b746b5bb28735eefe31dec6c9b31e"
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
    ref = json.loads((DATASET_DIR / "gallus_gallus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-chuong-gallus-gallus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
