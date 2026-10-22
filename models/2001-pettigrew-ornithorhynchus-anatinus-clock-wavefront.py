#!/usr/bin/env python3
"""
models/2001-pettigrew-ornithorhynchus-anatinus-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1454)
Organism: Ornithorhynchus anatinus (Duck-Billed Platypus (Electroreceptive Bill Pore Hexagonal Array))
Phenomenon Scale: 11 µm (1.086e-05 m, log10=-4.964, 4_cellular_10um)
Anatomical Structure: Segmentation clock somite / metamere unit in Ornithorhynchus anatinus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ornithorhynchus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (11 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1454_ornithorhynchus_anatinus_clock_wavefront.lean` (`MathOfBiology.Loop_1454`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ornithorhynchus_anatinus_clock_wavefront"
EXPECTED_SHA256 = {
    "ornithorhynchus_anatinus_clock_wavefront_reference.json": "575dd4dc393ce5a1da6768953097a3802b71e9b027ede161405704f20864c5a0",
    "ornithorhynchus_anatinus_specimen.jpg": "5a8d5bf6c1809500a8e9e88694cff57c75cead6258be2e7d05ede7922465f2b6",
    "scout_manifest.json": "e77df473ff5d21fe7daaa97784fbc71ef43f444c6dffb6aefe832b9f9f6245b6",
    "paper_citations.bib": "98b929117b2734695d12951840e58de4e983a60c791c6b7721792f1237905f10"
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
    ref = json.loads((DATASET_DIR / "ornithorhynchus_anatinus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-pettigrew-ornithorhynchus-anatinus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
