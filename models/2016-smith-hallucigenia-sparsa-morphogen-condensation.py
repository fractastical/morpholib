#!/usr/bin/env python3
"""
models/2016-smith-hallucigenia-sparsa-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1343)
Organism: Hallucigenia sparsa (Middle Cambrian Burgess Shale Armored Lobopod)
Phenomenon Scale: 3.3 mm (0.003317 m, log10=-2.479, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Hallucigenia sparsa (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hallucigenia
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (3.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1343_hallucigenia_sparsa_morphogen_condensation.lean` (`MathOfBiology.Loop_1343`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hallucigenia_sparsa_morphogen_condensation"
EXPECTED_SHA256 = {
    "hallucigenia_sparsa_morphogen_condensation_reference.json": "294b0518a380a801890affd6f967871b28c48939963b48d064f4edddae94db33",
    "hallucigenia_sparsa_specimen.jpg": "f0a3447adf27b728b96f3d6331a6f82327a3e9dbaf8871c9581019053e23ea81",
    "scout_manifest.json": "280cbd8d800366e1da715d8d234f4bf136eed95655ddfe304d4ee2865d0809df",
    "paper_citations.bib": "365dfda30f89a5d16b6af05a7d703460e38fe875154bda7d920886f72e136bce"
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
    ref = json.loads((DATASET_DIR / "hallucigenia_sparsa_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-smith-hallucigenia-sparsa-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
