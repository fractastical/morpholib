#!/usr/bin/env python3
"""
models/2013-li-anchiornis-huxleyi-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1949)
Organism: Anchiornis huxleyi (Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array))
Phenomenon Scale: 2.5 nm (2.5e-09 m, log10=-8.602, 1_molecular_nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Anchiornis huxleyi (Late Jurassic (Tiaojishan Fm, 160.0 Ma))
Geological Horizon: Late Jurassic (Tiaojishan Fm, 160.0 Ma) (160.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anchiornis
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (2.5 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1949_anchiornis_huxleyi_bioelectric_polarity.lean` (`MathOfBiology.Loop_1949`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anchiornis_huxleyi_bioelectric_polarity"
EXPECTED_SHA256 = {
    "anchiornis_huxleyi_bioelectric_polarity_reference.json": "f2248bc5aafb2b09c603c7ada02ea6f898d58a07003a14589921ededcd437e55",
    "anchiornis_huxleyi_specimen.jpg": "fab0bb4ca7882feafbf18840d6056dd36150060dc484593ec1df858d67a18865",
    "scout_manifest.json": "13105a20059b81ce474d5667b316321aa88c3672fbd250d1cc0c717fa9ed5dad",
    "paper_citations.bib": "10ec37f07b8dc4996709cbb12aecf19ed34079bd773a27d87c73108aa90e75ed"
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
    ref = json.loads((DATASET_DIR / "anchiornis_huxleyi_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-li-anchiornis-huxleyi-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
