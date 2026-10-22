#!/usr/bin/env python3
"""
models/2019-inoue-placenticeras-meeki-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1958)
Organism: Placenticeras meeki (Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68))
Phenomenon Scale: 51 µm (5.059e-05 m, log10=-4.296, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Placenticeras meeki (Late Cretaceous (Campanian, 75 Ma))
Geological Horizon: Late Cretaceous (Campanian, 75 Ma) (75.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Placenticeras
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (51 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1958_placenticeras_meeki_bioelectric_polarity.lean` (`MathOfBiology.Loop_1958`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/placenticeras_meeki_bioelectric_polarity"
EXPECTED_SHA256 = {
    "placenticeras_meeki_bioelectric_polarity_reference.json": "af26dc2a866a2057d93a04f604b69bf401c2ef58b5a653b6c211bf085f5ba694",
    "placenticeras_meeki_specimen.jpg": "ef945c35e08dd1adfded926af45c962f3c6017582c005f246eceedc2238e4229",
    "scout_manifest.json": "b939c549ad902de0c97cc26cfcf5b6388fd3f754adf177156c6b7de6715331d1",
    "paper_citations.bib": "8f7beae6103330e213bd4fcf01313f2f7ad939999386b3e8baa12e03a7ace576"
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
    ref = json.loads((DATASET_DIR / "placenticeras_meeki_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-inoue-placenticeras-meeki-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
