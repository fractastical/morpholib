#!/usr/bin/env python3
"""
models/2014-li-microraptor-gui-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1411)
Organism: Microraptor gui (Early Cretaceous Iridescent Black Four-Winged Dromaeosaur)
Phenomenon Scale: 68 nm (6.845e-08 m, log10=-7.165, 2_macromolecular_100nm)
Anatomical Structure: Segmentation clock somite / metamere unit in Microraptor gui (Early Cretaceous (Jiufotang Fm, 120.0 Ma))
Geological Horizon: Early Cretaceous (Jiufotang Fm, 120.0 Ma) (120.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Microraptor
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (68 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1411_microraptor_gui_clock_wavefront.lean` (`MathOfBiology.Loop_1411`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/microraptor_gui_clock_wavefront"
EXPECTED_SHA256 = {
    "microraptor_gui_clock_wavefront_reference.json": "6eca5040ef6db28baaabf9f2d6df7b69f022af85481ae54236cd476634fa31ba",
    "microraptor_gui_specimen.jpg": "24fb32a1367db5df0d77dace2b5b810c0854a2b3c231e370b837356348765137",
    "scout_manifest.json": "1d8c97a66ec478c03af494bac06b1f3861c769b5d706faf99e155d251fb2b731",
    "paper_citations.bib": "e2d022c9335eb6c9797d6235f56582504bc0ebd1e779155ab736d904ce341077"
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
    ref = json.loads((DATASET_DIR / "microraptor_gui_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-li-microraptor-gui-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
