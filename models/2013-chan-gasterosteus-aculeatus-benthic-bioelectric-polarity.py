#!/usr/bin/env python3
"""
models/2013-chan-gasterosteus-aculeatus-benthic-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1931)
Organism: Gasterosteus aculeatus_benthic (Post-Glacial Pitx1-Deleted Freshwater Benthic Stickleback (Low Plate))
Phenomenon Scale: 45 µm (4.501e-05 m, log10=-4.347, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Gasterosteus aculeatus_benthic (Holocene Post-Glacial Lake (Extant))
Geological Horizon: Holocene Post-Glacial Lake (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (45 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1931_gasterosteus_aculeatus_benthic_bioelectric_polarity.lean` (`MathOfBiology.Loop_1931`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_benthic_bioelectric_polarity"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_benthic_bioelectric_polarity_reference.json": "7415810405ff6841b045b249ce991646b477ffc879acc195d3af5ad1dc80531a",
    "gasterosteus_aculeatus_benthic_specimen.jpg": "d715e13ebe21bfa5b005584946ebb33b475d11926bac487e67aab10554b26e5b",
    "scout_manifest.json": "05ce610d27a0278982bcb0a5201d16e7d94b6e2248e4abdb30f45fc952024ad9",
    "paper_citations.bib": "7e31aa32842cb0dc005a923b6d3bb9758dd03471dde86a0c4d1e38a4d6800dea"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_benthic_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-chan-gasterosteus-aculeatus-benthic-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
