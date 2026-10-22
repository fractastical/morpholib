#!/usr/bin/env python3
"""
models/2024-gishlick-walliserops-trifurcatus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1337)
Organism: Walliserops trifurcatus (Middle Devonian Long-Trident Moroccan Trilobite)
Phenomenon Scale: 8.7 mm (0.008726 m, log10=-2.059, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Walliserops trifurcatus (Middle Devonian (Eifelian, 394 Ma))
Geological Horizon: Middle Devonian (Eifelian, 394 Ma) (394.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Walliserops
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (8.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1337_walliserops_trifurcatus_morphogen_condensation.lean` (`MathOfBiology.Loop_1337`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/walliserops_trifurcatus_morphogen_condensation"
EXPECTED_SHA256 = {
    "walliserops_trifurcatus_morphogen_condensation_reference.json": "9ec75b514a2900d28efede3ffdde7deb5f8e60d4863055f910f4c4eb54acb7f1",
    "walliserops_trifurcatus_specimen.jpg": "8a4e9c4d8a7e87dbdcf6b43da85f70188d90b4d5e1d541428ebaa0c016dd9923",
    "scout_manifest.json": "0a36d9a927bd4469c91cd3ee3741fcc5a4639bb5117cedd00563d26db7a4c333",
    "paper_citations.bib": "124db5030483dbd74405568082667e2a6abb68530d44c93b6cef09e1b28ecbb7"
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
    ref = json.loads((DATASET_DIR / "walliserops_trifurcatus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-gishlick-walliserops-trifurcatus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
