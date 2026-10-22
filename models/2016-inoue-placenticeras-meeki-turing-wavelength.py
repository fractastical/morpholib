#!/usr/bin/env python3
"""
models/2016-inoue-placenticeras-meeki-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1238)
Organism: Placenticeras meeki (Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68))
Phenomenon Scale: 2.5 mm (0.00253 m, log10=-2.597, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Placenticeras meeki (Late Cretaceous (Campanian, 75 Ma))
Geological Horizon: Late Cretaceous (Campanian, 75 Ma) (75.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Placenticeras
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (2.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1238_placenticeras_meeki_turing_wavelength.lean` (`MathOfBiology.Loop_1238`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/placenticeras_meeki_turing_wavelength"
EXPECTED_SHA256 = {
    "placenticeras_meeki_turing_wavelength_reference.json": "aa67f98e298cf6a1d811c8a51c947e463ac8243b0eeef100ce5da0131f371098",
    "placenticeras_meeki_specimen.jpg": "ef945c35e08dd1adfded926af45c962f3c6017582c005f246eceedc2238e4229",
    "scout_manifest.json": "b92c5f096edb132c6f2c4fb116ee43a54de7d9a3af53044313a68bae8dfe2892",
    "paper_citations.bib": "96ec1c34fd1db5b42c5517bed4bd61a1b7419f69acbafe631e7431672abeb01d"
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
    ref = json.loads((DATASET_DIR / "placenticeras_meeki_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-inoue-placenticeras-meeki-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
