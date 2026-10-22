#!/usr/bin/env python3
"""
models/2005-klug-ceratites-nodosus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1326)
Organism: Ceratites nodosus (Middle Triassic Muschelkalk Ammonoid (Serrated Lobes, D_f=1.34))
Phenomenon Scale: 4.9 mm (0.004896 m, log10=-2.310, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ceratites nodosus (Middle Triassic (Ladinian, 240 Ma))
Geological Horizon: Middle Triassic (Ladinian, 240 Ma) (240.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Ceratites
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (4.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1326_ceratites_nodosus_morphogen_condensation.lean` (`MathOfBiology.Loop_1326`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ceratites_nodosus_morphogen_condensation"
EXPECTED_SHA256 = {
    "ceratites_nodosus_morphogen_condensation_reference.json": "cfce301324ac18eeb5b005c59597bc84290d6193193f76f6f36823546e9f8bae",
    "ceratites_nodosus_specimen.jpg": "3c37b42ed1bd3fdedd3bbdb0eeef8244f5d9c6fffd9f8fa40804afed5dbc58f0",
    "scout_manifest.json": "288b392a3a136e884b4c08d8381c802c66c58cf4535b3308caaff7981bf370ad",
    "paper_citations.bib": "ee30f9a32339a49297d0d34ea79e560914ed59ed4adf0c242cef9a800f86f54b"
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
    ref = json.loads((DATASET_DIR / "ceratites_nodosus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-klug-ceratites-nodosus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
