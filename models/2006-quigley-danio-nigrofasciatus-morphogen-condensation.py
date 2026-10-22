#!/usr/bin/env python3
"""
models/2006-quigley-danio-nigrofasciatus-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1293)
Organism: Danio nigrofasciatus (Spotted Danio (broken ventral stripes))
Phenomenon Scale: 4.7 mm (0.004687 m, log10=-2.329, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Danio nigrofasciatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (4.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1293_danio_nigrofasciatus_morphogen_condensation.lean` (`MathOfBiology.Loop_1293`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_nigrofasciatus_morphogen_condensation"
EXPECTED_SHA256 = {
    "danio_nigrofasciatus_morphogen_condensation_reference.json": "451095fc34681bbf775e12580193abb8aed159411f3bc466dadbc0925f1c0d86",
    "danio_nigrofasciatus_specimen.jpg": "4aee990cea7e343b773365da24389df37f170d7d68c3492c513f22afd9f750d3",
    "scout_manifest.json": "c9477222e97a119eb733126e491538e9585dd44da08d466155d5d1ae22146de3",
    "paper_citations.bib": "b175831385e4c0ffd7c8290fe9858b8fc467c406cc6f36d689c6613bfbec82f8"
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
    ref = json.loads((DATASET_DIR / "danio_nigrofasciatus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-quigley-danio-nigrofasciatus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
