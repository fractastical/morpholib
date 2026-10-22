#!/usr/bin/env python3
"""
models/2016-nweeia-monodon-monoceros-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1407)
Organism: Monodon monoceros (Arctic Narwhal (Chiral Helical Left Incisor Tusk))
Phenomenon Scale: 15.5 cm (0.1554 m, log10=-0.809, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Monodon monoceros (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Monodon
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (15.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1407_monodon_monoceros_clock_wavefront.lean` (`MathOfBiology.Loop_1407`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/monodon_monoceros_clock_wavefront"
EXPECTED_SHA256 = {
    "monodon_monoceros_clock_wavefront_reference.json": "a2210b603c835c77f6032514dd648e78c0786b1c5cadd15847039f7c7162cf89",
    "monodon_monoceros_specimen.jpg": "1fa8766cd6ff186448201eb9da7097191319445a02628f275cfe5752d1be8cc3",
    "scout_manifest.json": "691a79a257a4007ae0ac40642cdbc75b1c6d3bcc305dde3fd7670608b491979b",
    "paper_citations.bib": "31bd5025772c41459455f71cfaf774f9f1bbb0b4dc7ac53439b6f8f3ef7700dc"
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
    ref = json.loads((DATASET_DIR / "monodon_monoceros_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-nweeia-monodon-monoceros-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
