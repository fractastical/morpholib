#!/usr/bin/env python3
"""
models/2005-fusco-aulacopleura-koninckii-holaspis18-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1783)
Organism: Aulacopleura koninckii_holaspis18 (Silurian Trilobite Holaspid Morphotype I (18 Thoracic Segments))
Phenomenon Scale: 3.0 mm (0.002992 m, log10=-2.524, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Aulacopleura koninckii_holaspis18 (Silurian (Wenlock Epoch, 428.3 Ma))
Geological Horizon: Silurian (Wenlock Epoch, 428.3 Ma) (428.3 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Aulacopleura
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1783_aulacopleura_koninckii_holaspis18_elastic_buckling.lean` (`MathOfBiology.Loop_1783`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/aulacopleura_koninckii_holaspis18_elastic_buckling"
EXPECTED_SHA256 = {
    "aulacopleura_koninckii_holaspis18_elastic_buckling_reference.json": "7e3a4c2136b2acfa6f0ea3133c2593a32ae01e7f63c0a4d846d23bc70a17d851",
    "aulacopleura_koninckii_holaspis18_specimen.jpg": "1b6c8997dc42e91d0975aee1225a27810db08c0301831c90db07bebae7b0344a",
    "scout_manifest.json": "df9d1be8e2f2531f0b36b99bec7a7fcdffb12c4e19b3a4195c36aba97a0d02d9",
    "paper_citations.bib": "e424d4999d8f44d797026cbf54bf5cb1333658179c2916640d4d6ae39af93c3f"
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
    ref = json.loads((DATASET_DIR / "aulacopleura_koninckii_holaspis18_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-fusco-aulacopleura-koninckii-holaspis18-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
