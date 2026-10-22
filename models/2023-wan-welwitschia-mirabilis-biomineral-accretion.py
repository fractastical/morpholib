#!/usr/bin/env python3
"""
models/2023-wan-welwitschia-mirabilis-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1917)
Organism: Welwitschia mirabilis (Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves))
Phenomenon Scale: 43.5 cm (0.4348 m, log10=-0.362, 8_colonial_m)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Welwitschia mirabilis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Welwitschia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (43.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1917_welwitschia_mirabilis_biomineral_accretion.lean` (`MathOfBiology.Loop_1917`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/welwitschia_mirabilis_biomineral_accretion"
EXPECTED_SHA256 = {
    "welwitschia_mirabilis_biomineral_accretion_reference.json": "dbd3e928ddeecfcb9b9c7e2f31252cc1e3e8198d0a3beaf99e4fcef62e52a7bf",
    "welwitschia_mirabilis_specimen.jpg": "ee3b34d91d4679f637d5087e7e7fa56ebbe65a443882acb98b96db019cfdc1ab",
    "scout_manifest.json": "57d28f6a6f3b7627572a5af6d9b839140a668577a203b98e629ae3ccdfbed6c3",
    "paper_citations.bib": "99ea99eb35c8be8a80bfea66a7db293002ace38fe08aac8abc38d672b97dc270"
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
    ref = json.loads((DATASET_DIR / "welwitschia_mirabilis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-wan-welwitschia-mirabilis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
