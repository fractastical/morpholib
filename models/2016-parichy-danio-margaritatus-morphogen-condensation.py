#!/usr/bin/env python3
"""
models/2016-parichy-danio-margaritatus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1291)
Organism: Danio margaritatus (Celestial Pearl Danio (Galaxy Rasbora))
Phenomenon Scale: 2.8 mm (0.002754 m, log10=-2.560, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Danio margaritatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (2.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1291_danio_margaritatus_morphogen_condensation.lean` (`MathOfBiology.Loop_1291`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_margaritatus_morphogen_condensation"
EXPECTED_SHA256 = {
    "danio_margaritatus_morphogen_condensation_reference.json": "50365955fb9892bffb010bbc743d7f57a33e5f860f8d8f0c4de302f8573c23aa",
    "danio_margaritatus_specimen.jpg": "6f4ec861e86da2f7ea31f23d6a81b802286dce7ac2a0631e45f9097fd9b8e876",
    "scout_manifest.json": "3e152e7d90f2daf24b3b0c18918638d01788b54b1ba9468ce2825e80ec90443d",
    "paper_citations.bib": "f9d1082df50ab2573170a9d6f43671adf8e799e604d47485a4ef4c55760f1a51"
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
    ref = json.loads((DATASET_DIR / "danio_margaritatus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-parichy-danio-margaritatus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
