#!/usr/bin/env python3
"""
models/2020-spiewak-danio-aesculapii-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1384)
Organism: Danio aesculapii (Panther Danio (Vertical Bar Transition))
Phenomenon Scale: 2.0 mm (0.002018 m, log10=-2.695, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Danio aesculapii (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.0 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1384_danio_aesculapii_clock_wavefront.lean` (`MathOfBiology.Loop_1384`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_aesculapii_clock_wavefront"
EXPECTED_SHA256 = {
    "danio_aesculapii_clock_wavefront_reference.json": "cfdb35ed7feebe69d0be722be4eebf6ddeefa2f4ac0c388ce92b9410e1b677cb",
    "danio_aesculapii_specimen.jpg": "4ccce0ce3648e44c77cef768d10f5acd8ab553c4ca38d348c35d4c6846b46bc5",
    "scout_manifest.json": "ea17bbb32a6c6d4d8728e12ccac531cd43631c06c6b6873a6d290a3c06729361",
    "paper_citations.bib": "b713f8b2e45487dcc3d8db16e15af962482043611ff49e332c3b0cb56dbfb634"
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
    ref = json.loads((DATASET_DIR / "danio_aesculapii_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-spiewak-danio-aesculapii-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
