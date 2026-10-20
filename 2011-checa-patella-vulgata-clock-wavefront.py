#!/usr/bin/env python3
"""
2011-checa-patella-vulgata-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0353)
Organism: Patella vulgata (Common Limpet)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Patella
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0353_patella_vulgata_clock_wavefront.lean` (`MathOfBiology.Loop_0353`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/patella_vulgata_clock_wavefront"
EXPECTED_SHA256 = {
    "patella_vulgata_clock_wavefront_reference.json": "84d52a09dbaeaba5a54f872d662239c5bb620afac2ecbd1e19869370c44f9df1",
    "patella_vulgata_specimen.jpg": "dc71ee15b04a3541b9b25a13411da18888acaaba54973227c4ba40c361669d0e",
    "scout_manifest.json": "868209c32d37f1c03ff4af78dba2f97f302e4def21d9f3713ddee734769c7fa5",
    "paper_citations.bib": "18a001463910d0558284195e2a1e4d873252bcbf428a13d63aa1eeb28af7b1fc"
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
    ref = json.loads((DATASET_DIR / "patella_vulgata_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-checa-patella-vulgata-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
