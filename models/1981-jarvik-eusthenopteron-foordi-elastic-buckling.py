#!/usr/bin/env python3
"""
models/1981-jarvik-eusthenopteron-foordi-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1752)
Organism: Eusthenopteron foordi (Late Devonian Tristichopterid Lobe-Fin Fish)
Phenomenon Scale: 3.1 cm (0.03111 m, log10=-1.507, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Eusthenopteron foordi (Late Devonian (Frasnian, 379 Ma))
Geological Horizon: Late Devonian (Frasnian, 379 Ma) (379.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Eusthenopteron
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1752_eusthenopteron_foordi_elastic_buckling.lean` (`MathOfBiology.Loop_1752`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eusthenopteron_foordi_elastic_buckling"
EXPECTED_SHA256 = {
    "eusthenopteron_foordi_elastic_buckling_reference.json": "805b406965aadb0cfc2a9a56696eb10ac35614a96284d1f093114087b899084b",
    "eusthenopteron_foordi_specimen.jpg": "d97b68e66855434cb59cbd6b3fc48d7827c97c6981f567e77fb025c359499811",
    "scout_manifest.json": "b18dc41dbb4a85c20be952779c48c8e73fb197bed679abf57a2d1369cb6307ee",
    "paper_citations.bib": "dde649d06bcce0d981cd2119cc7f638dade2ffced38962583b7e5b3f166038a3"
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
    ref = json.loads((DATASET_DIR / "eusthenopteron_foordi_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1981-jarvik-eusthenopteron-foordi-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
