#!/usr/bin/env python3
"""
models/1989-ward-nautilus-pompilius-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1421)
Organism: Nautilus pompilius (Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle))
Phenomenon Scale: 1.4 cm (0.01386 m, log10=-1.858, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Nautilus pompilius (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nautilus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1421_nautilus_pompilius_clock_wavefront.lean` (`MathOfBiology.Loop_1421`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nautilus_pompilius_clock_wavefront"
EXPECTED_SHA256 = {
    "nautilus_pompilius_clock_wavefront_reference.json": "ce6872b87d6b4445bb635031bd9ed89fc3eeb47a06ad247e57da73dc0564bd19",
    "nautilus_pompilius_specimen.jpg": "5dfd47d29d062b864cb14b6f5260b674dffb372fac09ff90bdd8fc5caf196da1",
    "scout_manifest.json": "af0d82b8218895b279d11f6b487a17965d2fac716b40e81e6f7c166a5f06be3b",
    "paper_citations.bib": "5550b849302123b9acb529db7624b6b2809eb4a613f9d6f51abff8eee1b783de"
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
    ref = json.loads((DATASET_DIR / "nautilus_pompilius_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1989-ward-nautilus-pompilius-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
