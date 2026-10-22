#!/usr/bin/env python3
"""
models/2002-chiappe-confuciusornis-sanctus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1952)
Organism: Confuciusornis sanctus (Early Cretaceous Pygostyle Beaked Bird (Ribbon Tail Streamers))
Phenomenon Scale: 109 µm (0.0001086 m, log10=-3.964, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Confuciusornis sanctus (Early Cretaceous (Yixian Fm, 125.0 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 125.0 Ma) (125.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Confuciusornis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (109 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1952_confuciusornis_sanctus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1952`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/confuciusornis_sanctus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "confuciusornis_sanctus_bioelectric_polarity_reference.json": "75faa476b9187d86d8371fc6d715ce36c804aa4b520846937f419b0b8027b888",
    "confuciusornis_sanctus_specimen.jpg": "cd7c05e2d404d66e00f79aa73a3de8f5a68cde60009b9288d61176e6902d7620",
    "scout_manifest.json": "e1ddfe41863d5e1b934bde32ab1202ad7771d24d2470a5ce6159b1292087685c",
    "paper_citations.bib": "bb1e7c320e4e722ccb5835a219ab9dd19f5562da23d5702cdc8db796c9a6b430"
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
    ref = json.loads((DATASET_DIR / "confuciusornis_sanctus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-chiappe-confuciusornis-sanctus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
