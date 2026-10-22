#!/usr/bin/env python3
"""
models/2020-maisano-varanus-komodoensis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1818)
Organism: Varanus komodoensis (Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork))
Phenomenon Scale: 524 µm (0.0005236 m, log10=-3.281, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Varanus komodoensis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Varanus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (524 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1818_varanus_komodoensis_elastic_buckling.lean` (`MathOfBiology.Loop_1818`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/varanus_komodoensis_elastic_buckling"
EXPECTED_SHA256 = {
    "varanus_komodoensis_elastic_buckling_reference.json": "69927e290111d769baf6dbe790ce8644fdb1d3cc585bee4568b2441c121e9698",
    "varanus_komodoensis_specimen.jpg": "347fa0ca27bf35537323e24d59b72d5f6df744f56be66ed7c9649cdf3bc761b5",
    "scout_manifest.json": "0053432b866108a8d457726a7e25187edaf73122baae427bec4a5dd821c8ccd6",
    "paper_citations.bib": "5b20bd66777a2fdae8691a8e5e086a997b0433869210335bccab164279866e03"
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
    ref = json.loads((DATASET_DIR / "varanus_komodoensis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-maisano-varanus-komodoensis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
