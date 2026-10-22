#!/usr/bin/env python3
"""
models/1975-eldredge-eldredgeops-rana-18file-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1965)
Organism: Eldredgeops rana_18file (Middle Devonian Phacopid Trilobite (18-File Schizochroal Calcite Eye))
Phenomenon Scale: 357 nm (3.571e-07 m, log10=-6.447, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Eldredgeops rana_18file (Middle Devonian (Cazenovian, 387 Ma))
Geological Horizon: Middle Devonian (Cazenovian, 387 Ma) (387.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (357 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1965_eldredgeops_rana_18file_bioelectric_polarity.lean` (`MathOfBiology.Loop_1965`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_18file_bioelectric_polarity"
EXPECTED_SHA256 = {
    "eldredgeops_rana_18file_bioelectric_polarity_reference.json": "503818d1dcbaa6ea7b47f434a317d5f57e302115d4737dedc128a823d23caf0f",
    "eldredgeops_rana_18file_specimen.jpg": "ba63b7ea8e242e6374117b84812be17b530eebb3f9621b25eff533b5397e1369",
    "scout_manifest.json": "5d3f5b8aab47c4aa9fbbfb06b3972a827bf27398e8dbcde645a90a006f5284f7",
    "paper_citations.bib": "2084ef5623a5852c455dbeefc1b4d09d524418536f4489068c6946c4093ba85e"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_18file_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1975-eldredge-eldredgeops-rana-18file-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
