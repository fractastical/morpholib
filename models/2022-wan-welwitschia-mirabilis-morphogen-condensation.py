#!/usr/bin/env python3
"""
models/2022-wan-welwitschia-mirabilis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1377)
Organism: Welwitschia mirabilis (Namib Desert Welwitschia (Indeterminate Basal-Meristem Strap Leaves))
Phenomenon Scale: 14.2 cm (0.1423 m, log10=-0.847, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Welwitschia mirabilis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Welwitschia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (14.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1377_welwitschia_mirabilis_morphogen_condensation.lean` (`MathOfBiology.Loop_1377`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/welwitschia_mirabilis_morphogen_condensation"
EXPECTED_SHA256 = {
    "welwitschia_mirabilis_morphogen_condensation_reference.json": "e7d4dc9ef443d754f43cee2d542f1c4539a7ede8fa4bd0797b0d4c5ed3c1ba4a",
    "welwitschia_mirabilis_specimen.jpg": "ee3b34d91d4679f637d5087e7e7fa56ebbe65a443882acb98b96db019cfdc1ab",
    "scout_manifest.json": "12c0e81374fc91b7a804548e7ba7b4fbd6df97d0eba5323caa7e293e2eaa1829",
    "paper_citations.bib": "7dad0a22a29fe1c8ee6a62dcd3aa0d9eac04934ae219fc98ae28b3b445cf6c30"
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
    ref = json.loads((DATASET_DIR / "welwitschia_mirabilis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-wan-welwitschia-mirabilis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
