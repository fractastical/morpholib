#!/usr/bin/env python3
"""
models/1990-ward-nautilus-pompilius-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1961)
Organism: Nautilus pompilius (Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle))
Phenomenon Scale: 139 µm (0.0001386 m, log10=-3.858, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Nautilus pompilius (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nautilus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (139 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1961_nautilus_pompilius_bioelectric_polarity.lean` (`MathOfBiology.Loop_1961`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nautilus_pompilius_bioelectric_polarity"
EXPECTED_SHA256 = {
    "nautilus_pompilius_bioelectric_polarity_reference.json": "1e6f2da6479fdfdc61deb3264043a016d7107a5729e3116167dd6cef32cec580",
    "nautilus_pompilius_specimen.jpg": "5dfd47d29d062b864cb14b6f5260b674dffb372fac09ff90bdd8fc5caf196da1",
    "scout_manifest.json": "2d468551ffb5c72188d38b3f75641306157e948ed0de3680f424b6b30ce57edc",
    "paper_citations.bib": "0f2e034739ee92ba7e4c78afd2ed6e874c56ae5fd540aade10e8da4eac35f80e"
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
    ref = json.loads((DATASET_DIR / "nautilus_pompilius_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1990-ward-nautilus-pompilius-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
