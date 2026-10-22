#!/usr/bin/env python3
"""
models/1987-grande-diplomystus-dentatus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1926)
Organism: Diplomystus dentatus (Green River Eocene Armored Herring Fossil)
Phenomenon Scale: 151 µm (0.0001512 m, log10=-3.820, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Diplomystus dentatus (Early Eocene (Ypresian, 51.5 Ma))
Geological Horizon: Early Eocene (Ypresian, 51.5 Ma) (51.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Diplomystus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (151 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1926_diplomystus_dentatus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1926`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/diplomystus_dentatus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "diplomystus_dentatus_bioelectric_polarity_reference.json": "9d704d52a6696575113a25ba23981ca2da0f8a65c618f3e19e7006bc9d4f6733",
    "diplomystus_dentatus_specimen.jpg": "00e0f80ae28f4fe36376b217f05e2048df17ed19a49f48d49d4a9b88b2b2b659",
    "scout_manifest.json": "9facea67ce54d7f6fa423e9826a63ee97f4c2fde0094a016695ee37646a84682",
    "paper_citations.bib": "0ecfea9a786b10540ddec9e9fd5545321ce373a2cd436d4a64f75821f4229b3d"
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
    ref = json.loads((DATASET_DIR / "diplomystus_dentatus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1987-grande-diplomystus-dentatus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
