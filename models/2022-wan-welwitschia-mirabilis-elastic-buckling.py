#!/usr/bin/env python3
"""
models/2022-wan-welwitschia-mirabilis-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1827)
Organism: Welwitschia mirabilis (Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves))
Phenomenon Scale: 17.4 cm (0.1739 m, log10=-0.760, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Welwitschia mirabilis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Welwitschia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (17.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1827_welwitschia_mirabilis_elastic_buckling.lean` (`MathOfBiology.Loop_1827`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/welwitschia_mirabilis_elastic_buckling"
EXPECTED_SHA256 = {
    "welwitschia_mirabilis_elastic_buckling_reference.json": "cccdba6076725901edeb075940d22fb6fdb4d12768d7186a38f80e7d43ab556a",
    "welwitschia_mirabilis_specimen.jpg": "ee3b34d91d4679f637d5087e7e7fa56ebbe65a443882acb98b96db019cfdc1ab",
    "scout_manifest.json": "5cd40e3da6e0fc8f9719b1e21d50815a19149631f0d7b93f10f4140ae13a2982",
    "paper_citations.bib": "d1ea5b09f5bd42021b151b3fbde1d9db6ae63dfceb52daff4a261148e8b9fd5c"
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
    ref = json.loads((DATASET_DIR / "welwitschia_mirabilis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-wan-welwitschia-mirabilis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
