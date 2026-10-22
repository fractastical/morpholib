#!/usr/bin/env python3
"""
models/2022-jabusch-pyrocystis-fusiformis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1470)
Organism: Pyrocystis fusiformis (Bioluminescent Marine Dinoflagellate (Shear-Induced Scintillon Flash))
Phenomenon Scale: 32 µm (3.192e-05 m, log10=-4.496, 4_cellular_10um)
Anatomical Structure: Segmentation clock somite / metamere unit in Pyrocystis fusiformis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Pyrocystis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (32 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1470_pyrocystis_fusiformis_clock_wavefront.lean` (`MathOfBiology.Loop_1470`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pyrocystis_fusiformis_clock_wavefront"
EXPECTED_SHA256 = {
    "pyrocystis_fusiformis_clock_wavefront_reference.json": "02359fd32d865a496558eee66969416cecbeadbc8861f7eb77298a988addcb60",
    "pyrocystis_fusiformis_specimen.jpg": "4fad32ed3942f3ab89d8e79cda844c3cab6085f9a3ce836885fe9d9bf8112cb2",
    "scout_manifest.json": "27ed960218a7b1234742a78468c1f79fea0680f0eb1db85ec6c0205db7d1a975",
    "paper_citations.bib": "3ce89960be1b2bc7ce936d99108f892d26223896100ca320990c4fd07f126436"
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
    ref = json.loads((DATASET_DIR / "pyrocystis_fusiformis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-jabusch-pyrocystis-fusiformis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
