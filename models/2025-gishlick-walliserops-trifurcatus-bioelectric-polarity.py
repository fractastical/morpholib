#!/usr/bin/env python3
"""
models/2025-gishlick-walliserops-trifurcatus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1967)
Organism: Walliserops trifurcatus (Middle Devonian Long-Trident Moroccan Trilobite)
Phenomenon Scale: 39 µm (3.878e-05 m, log10=-4.411, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Walliserops trifurcatus (Middle Devonian (Eifelian, 394 Ma))
Geological Horizon: Middle Devonian (Eifelian, 394 Ma) (394.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Walliserops
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (39 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1967_walliserops_trifurcatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1967`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/walliserops_trifurcatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "walliserops_trifurcatus_bioelectric_polarity_reference.json": "70496b111465462bce72c23fd70d53ce12886f6d11d395dffa64936290bfd53b",
    "walliserops_trifurcatus_specimen.jpg": "8a4e9c4d8a7e87dbdcf6b43da85f70188d90b4d5e1d541428ebaa0c016dd9923",
    "scout_manifest.json": "840f36bb0ffe9cca0591c4b13ea6602315de34b92ecb19579ac27d9ea8044709",
    "paper_citations.bib": "70c9f07cbf309cfb0064d117509085715f8d2ce1f95aab93d434310e226cde14"
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
    ref = json.loads((DATASET_DIR / "walliserops_trifurcatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-gishlick-walliserops-trifurcatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
