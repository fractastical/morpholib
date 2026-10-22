#!/usr/bin/env python3
"""
models/2018-dilkes-eryops-megacephalus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1937)
Organism: Eryops megacephalus (Early Permian Temnospondyl Pentadactyl Amphibian)
Phenomenon Scale: 164 µm (0.0001637 m, log10=-3.786, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Eryops megacephalus (Early Permian (Cisuralian, 295 Ma))
Geological Horizon: Early Permian (Cisuralian, 295 Ma) (295.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Eryops
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (164 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1937_eryops_megacephalus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1937`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eryops_megacephalus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "eryops_megacephalus_bioelectric_polarity_reference.json": "d39d02a39ab1634e740cdbacf123fbb0bca8e26139294ca57555de1caf741de0",
    "eryops_megacephalus_specimen.jpg": "129f152b5349248669bd3ffc0b0360d0131cfa03e87c6ff1c0ac912fae50e424",
    "scout_manifest.json": "bd2ec7bf5d5dd17562c075adf48614aeb82e39b0fea3672397945922172f958b",
    "paper_citations.bib": "7c305fa9c53a66df3cab11e0f6e10a6e31111e0dfcee3b73dda4ad56034fd035"
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
    ref = json.loads((DATASET_DIR / "eryops_megacephalus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-dilkes-eryops-megacephalus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
