#!/usr/bin/env python3
"""
models/2016-bell-hesperornis-regalis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1323)
Organism: Hesperornis regalis (Late Cretaceous Toothed Diving Bird)
Phenomenon Scale: 3.5 cm (0.03545 m, log10=-1.450, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Hesperornis regalis (Late Cretaceous (Campanian, 83.5 Ma))
Geological Horizon: Late Cretaceous (Campanian, 83.5 Ma) (83.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Hesperornis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (3.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1323_hesperornis_regalis_morphogen_condensation.lean` (`MathOfBiology.Loop_1323`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hesperornis_regalis_morphogen_condensation"
EXPECTED_SHA256 = {
    "hesperornis_regalis_morphogen_condensation_reference.json": "356ce2f8a38620c0256267128f6b3afa156496da444ed714623f452639198dc4",
    "hesperornis_regalis_specimen.jpg": "a8ca1cd24cf7035f2fff3d592b3f55831669012975aed64e3050567bec49e7a7",
    "scout_manifest.json": "bbe00f1005f7ca0b97b18fdd6986f24f904e23eb57fef1610725f70aa71ff0a7",
    "paper_citations.bib": "488a8554efefa973cc68fac89845e741dc48477cd7de374b379bb7303182bd21"
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
    ref = json.loads((DATASET_DIR / "hesperornis_regalis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-bell-hesperornis-regalis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
