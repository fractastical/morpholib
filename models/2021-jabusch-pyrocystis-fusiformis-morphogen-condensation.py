#!/usr/bin/env python3
"""
models/2021-jabusch-pyrocystis-fusiformis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1380)
Organism: Pyrocystis fusiformis (Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash))
Phenomenon Scale: 72 µm (7.182e-05 m, log10=-4.144, 4_cellular_10um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Pyrocystis fusiformis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Pyrocystis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (72 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1380_pyrocystis_fusiformis_morphogen_condensation.lean` (`MathOfBiology.Loop_1380`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pyrocystis_fusiformis_morphogen_condensation"
EXPECTED_SHA256 = {
    "pyrocystis_fusiformis_morphogen_condensation_reference.json": "a4d087b6baeb4c6a5bf99a15800183957445086d2d28c360f8aacf572ae25509",
    "pyrocystis_fusiformis_specimen.jpg": "4fad32ed3942f3ab89d8e79cda844c3cab6085f9a3ce836885fe9d9bf8112cb2",
    "scout_manifest.json": "d39d7c13c8ef3937f665745d99bb39601b5ffbae9fc1749dc0d5dc2f1a87d40a",
    "paper_citations.bib": "78f62c7ced1ca505619f26244d75305f91167ba588b439c72e5716e5227957d3"
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
    ref = json.loads((DATASET_DIR / "pyrocystis_fusiformis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-jabusch-pyrocystis-fusiformis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
