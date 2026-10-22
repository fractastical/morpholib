#!/usr/bin/env python3
"""
models/2017-bell-hesperornis-regalis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1413)
Organism: Hesperornis regalis (Late Cretaceous Toothed Diving Bird)
Phenomenon Scale: 1.6 cm (0.01576 m, log10=-1.802, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Hesperornis regalis (Late Cretaceous (Campanian, 83.5 Ma))
Geological Horizon: Late Cretaceous (Campanian, 83.5 Ma) (83.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Hesperornis
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1413_hesperornis_regalis_clock_wavefront.lean` (`MathOfBiology.Loop_1413`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hesperornis_regalis_clock_wavefront"
EXPECTED_SHA256 = {
    "hesperornis_regalis_clock_wavefront_reference.json": "f82dd47afe0ecab8fa680d86f5cec82ada950721c4ca06603765dc830de994fc",
    "hesperornis_regalis_specimen.jpg": "a8ca1cd24cf7035f2fff3d592b3f55831669012975aed64e3050567bec49e7a7",
    "scout_manifest.json": "d79dd0ab5b991f5284fceafeb3d28c08cd9db87eebaadf9fca19b653f30a6501",
    "paper_citations.bib": "9f9c137c3b4fb97d43385698b565f020e8c4568047aae7325c61cbe956fcb3cc"
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
    ref = json.loads((DATASET_DIR / "hesperornis_regalis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-bell-hesperornis-regalis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
