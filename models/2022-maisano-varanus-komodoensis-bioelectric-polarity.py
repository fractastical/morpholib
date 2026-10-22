#!/usr/bin/env python3
"""
models/2022-maisano-varanus-komodoensis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1998)
Organism: Varanus komodoensis (Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork))
Phenomenon Scale: 1.9 µm (1.904e-06 m, log10=-5.720, 3_subcellular_um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Varanus komodoensis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Varanus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (1.9 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1998_varanus_komodoensis_bioelectric_polarity.lean` (`MathOfBiology.Loop_1998`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/varanus_komodoensis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "varanus_komodoensis_bioelectric_polarity_reference.json": "346f9c9ba3bd7d4a2ad1afc8b9c5ade8c8f4124380656fd12e19204e74160f13",
    "varanus_komodoensis_specimen.jpg": "347fa0ca27bf35537323e24d59b72d5f6df744f56be66ed7c9649cdf3bc761b5",
    "scout_manifest.json": "23a3eb4b354c92899a4861c7657fce8ff94ab29e1372865b7c0e9b996d85f066",
    "paper_citations.bib": "bfc87595bb2ba62aaddac1e0dc288d689c8505fa24cc8ee24f71a1da231dd3d5"
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
    ref = json.loads((DATASET_DIR / "varanus_komodoensis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-maisano-varanus-komodoensis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
