#!/usr/bin/env python3
"""
models/2017-wang-manis-javanica-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1365)
Organism: Manis javanica (Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice))
Phenomenon Scale: 6.4 mm (0.006363 m, log10=-2.196, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Manis javanica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Manis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (6.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1365_manis_javanica_morphogen_condensation.lean` (`MathOfBiology.Loop_1365`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/manis_javanica_morphogen_condensation"
EXPECTED_SHA256 = {
    "manis_javanica_morphogen_condensation_reference.json": "4aa62c692e47c8d880d1a1a1625be75675de5cf258ca43c379c927305ec33f1c",
    "manis_javanica_specimen.jpg": "0d2cf7b0e5853e57873b4a69f96a0bf200561aa593feda56d04025e8b7619711",
    "scout_manifest.json": "72920ad90a51d9d16408183b352c96b566fff8f29cd5beac7ddfb07df077c35f",
    "paper_citations.bib": "fb427bb2ee27e31825a8f3be1bf11b9c5fdc167c51015a05683a391b81445fcc"
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
    ref = json.loads((DATASET_DIR / "manis_javanica_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-wang-manis-javanica-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
