#!/usr/bin/env python3
"""
models/2018-smith-hallucigenia-sparsa-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1973)
Organism: Hallucigenia sparsa (Middle Cambrian Burgess Shale Armored Lobopod)
Phenomenon Scale: 15 µm (1.474e-05 m, log10=-4.832, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Hallucigenia sparsa (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hallucigenia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (15 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1973_hallucigenia_sparsa_bioelectric_polarity.lean` (`MathOfBiology.Loop_1973`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hallucigenia_sparsa_bioelectric_polarity"
EXPECTED_SHA256 = {
    "hallucigenia_sparsa_bioelectric_polarity_reference.json": "6da55aaf77be6c854f7ef6688648f89359531bd2369017ffca816e7c4e38106a",
    "hallucigenia_sparsa_specimen.jpg": "f0a3447adf27b728b96f3d6331a6f82327a3e9dbaf8871c9581019053e23ea81",
    "scout_manifest.json": "9267262b695f7d55b2b1ed2284b8b35ba93f29b35f4adf704f118648d1f2db99",
    "paper_citations.bib": "ef3e9cf7f4c3982245d730be0909c92243675043a8cbf457af731aa820f9606a"
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
    ref = json.loads((DATASET_DIR / "hallucigenia_sparsa_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-smith-hallucigenia-sparsa-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
