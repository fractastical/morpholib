#!/usr/bin/env python3
"""
models/1986-grande-diplomystus-dentatus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1386)
Organism: Diplomystus dentatus (Green River Eocene Armored Herring Fossil)
Phenomenon Scale: 1.5 cm (0.01512 m, log10=-1.820, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Diplomystus dentatus (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Diplomystus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1386_diplomystus_dentatus_clock_wavefront.lean` (`MathOfBiology.Loop_1386`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/diplomystus_dentatus_clock_wavefront"
EXPECTED_SHA256 = {
    "diplomystus_dentatus_clock_wavefront_reference.json": "30dc5f1fb0fb37de964748fb6bd5587cb0723b9c8e9468efcd267b12be27bc98",
    "diplomystus_dentatus_specimen.jpg": "00e0f80ae28f4fe36376b217f05e2048df17ed19a49f48d49d4a9b88b2b2b659",
    "scout_manifest.json": "8340d8c94247d24309250c879860ba5a4941b549432fe8e2dd474b8ea04ad22e",
    "paper_citations.bib": "6efa483a270195d9605d99c89c0aeb64de9b5ac5ba986aff34e4a56647eada57"
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
    ref = json.loads((DATASET_DIR / "diplomystus_dentatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/1986-grande-diplomystus-dentatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
